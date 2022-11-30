import os 
import json 
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import pickle as pkl

from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.cluster import KMeans
import numpy as np

from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup



from common.metrics import ClusteringMetricsWrapper, PsuedoLabelMetricWrapper
from common.predictions import ClusterPredictionsWrapper
from .RoCORE_layers import ZeroShotModel, L2Reg, compute_kld

import common.log as log
logger = log.get_logger('root')

class RoCOREModel(pl.LightningModule):
    def __init__(self, args, tokenizer, train_len:int = 1000) -> None:
        super().__init__()
        self.config = args  
        
        self.tokenizer = tokenizer 
        self.model_config = AutoConfig.from_pretrained(args.model_name_or_path, output_hidden_states = True)
        pretrained_model = AutoModel.from_pretrained(args.model_name_or_path, config = self.model_config)
        embeddings = pretrained_model.resize_token_embeddings(len(self.tokenizer))
        self.train_len=train_len # this is required to set up the optimizer 

        self.net = ZeroShotModel(args, args.known_types, args.unknown_types, self.model_config, pretrained_model, unfreeze_layers = [args.layer])
        self.train_unknown_metrics_wrapper = ClusteringMetricsWrapper(stage='train',
            known=False,
            prefix='train_unknown', known_classes=0) # no shift 
        self.val_unknown_metrics_wrapper = ClusteringMetricsWrapper(stage='val',
            known=False, 
            prefix='val_unknown',  known_classes=0)
        self.test_unknown_metrics_wrapper = ClusteringMetricsWrapper(stage='test',
            known=False,
            prefix='test_unknown',  known_classes=0) 
        
        self.train_known_metrics_wrapper = ClusteringMetricsWrapper(stage='train',
            known=True,
            prefix='train_known', 
            known_classes=args.known_types)
        self.val_known_metrics_wrapper = ClusteringMetricsWrapper(stage='val',
            known=True,
            prefix='val_known', known_classes=args.known_types)
        self.test_known_metrics_wrapper = ClusteringMetricsWrapper(stage='test', 
            known=True,
            prefix='test_known',  known_classes=args.known_types)

        if args.eval_only:
            self.predictions_wrapper = ClusterPredictionsWrapper(reassign=True, prefix='test_unknown', 
            known_classes=0)

    def forward(self, inputs):
        pass 
    
    def on_validation_epoch_start(self) -> None:
        # reset all metrics
        self.train_unknown_metrics_wrapper.reset() 
        self.val_unknown_metrics_wrapper.reset() 
        self.test_unknown_metrics_wrapper.reset()

        self.train_known_metrics_wrapper.reset() 
        self.val_known_metrics_wrapper.reset() 
        self.test_known_metrics_wrapper.reset() 

        return 

    def on_train_epoch_start(self): 
        logger.info('updating cluster centers....')
        train_dl = self.trainer.train_dataloader.loaders
        known_centers = torch.zeros(self.config.known_types, self.config.kmeans_dim, device = self.device)
        num_samples = [0] * self.config.known_types 
        with torch.no_grad():
            uid2pl = {} # pseudo labels 
            unknown_uid_list = [] 
            unknown_vec_list = []
            seen_uid = set() # oversampling for the unknown part, so we remove them here 
            for batch in tqdm(iter(train_dl)):
                labels = batch[0]['labels']
                known_mask = batch[0]['known_mask']
                metadata = batch[0]['meta']
                batch_size = len(metadata) 

                # move batch to gpu 
                for key in ['token_ids', 'attn_mask','head_spans','tail_spans']:
                    batch[0][key] = batch[0][key].to(self.device)
                
                commonspace_rep = self.net.forward(batch[0], msg = 'similarity') # (batch_size, hidden_dim)
                for i in range(batch_size):
                    if known_mask[i] == True:
                        l = labels[i]
                        known_centers[l] += commonspace_rep[i]
                        num_samples[l] += 1 
                    else:
                        uid = metadata[i]['uid']
                        if uid not in seen_uid:
                            seen_uid.add(uid)
                            unknown_uid_list.append(uid)
                            unknown_vec_list.append(commonspace_rep[i].cpu().numpy())
            
            # cluster unknown classes 
            clf = KMeans(n_clusters=self.config.unknown_types,random_state=0,algorithm='full')
            rep = np.stack(unknown_vec_list, axis=0)  
            label_pred = clf.fit_predict(rep)# from 0 to args.new_class - 1
            self.net.ct_loss_u.centers = torch.from_numpy(clf.cluster_centers_).to(self.device)# (num_class, kmeans_dim)
            for i in range(len(unknown_vec_list)): 
                uid = unknown_uid_list[i]
                pseudo = label_pred[i]
                uid2pl[uid] = pseudo + self.config.known_types 
                
            
            train_dl.dataset.update_pseudo_labels(uid2pl) 
            logger.info('updating pseudo labels...')
            pl_acc = train_dl.dataset.check_pl_acc() 
            self.log('train/pl_acc', pl_acc, on_epoch=True)
            

            # update center for known types 
            for c in range(self.config.known_types):
                known_centers[c] /= num_samples[c]
            self.net.ct_loss_l.centers = known_centers
        return 
    



    def _compute_unknown_margin_loss(self, batch: Dict[str, torch.Tensor], pseudo_labels: torch.LongTensor, known_mask: torch.BoolTensor) -> torch.FloatTensor:
        # convert 1d pseudo label into 2d pairwise pseudo label 
        assert (torch.min(pseudo_labels) >= self.config.known_types) 
        
        pair_label = (pseudo_labels.unsqueeze(0) == pseudo_labels.unsqueeze(1)).float()
        logits = self.net.forward(batch, mask=~known_mask, msg = 'unlabeled') # (batch_size, new_class)
        # this only predicts over new classes 
        unknown_batch_size = pseudo_labels.size(0)
        expanded_logits = logits.expand(unknown_batch_size, -1, -1)
        expanded_logits2 = expanded_logits.transpose(0, 1)
        kl1 = compute_kld(expanded_logits.detach(), expanded_logits2)
        kl2 = compute_kld(expanded_logits2.detach(), expanded_logits) # (batch_size, batch_size)
        assert kl1.requires_grad
        unknown_class_loss = torch.mean(pair_label * (kl1 + kl2) + (1 - pair_label) * (torch.relu(self.config.sigmoid - kl1) + torch.relu(self.config.sigmoid - kl2)))
        return unknown_class_loss


    def _compute_reconstruction_loss(self, batch: Dict[str, torch.Tensor], known_mask: torch.BoolTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        # recon loss for known classes 
        commonspace_rep_known, rec_loss_known = self.net.forward(batch, mask=known_mask,  msg = 'reconstruct') # (batch_size, kmeans_dim)
        # recon loss for unknown classes 
        _ , rec_loss_unknown = self.net.forward(batch,mask=~known_mask, msg = 'reconstruct') # (batch_size, kmeans_dim)
        reconstruction_loss = (rec_loss_known.mean() + rec_loss_unknown.mean()) / 2
        # center loss for known classes 
        center_loss = self.config.center_loss * self.net.ct_loss_l(labels[known_mask], commonspace_rep_known)
        l2_reg = 1e-5 * (L2Reg(self.net.similarity_encoder) + L2Reg(self.net.similarity_decoder))
        loss = reconstruction_loss + center_loss + l2_reg 
        return loss 


    def _compute_ce_loss(self, batch: Dict[str, torch.Tensor], known_mask: torch.BoolTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        '''
        Cross entropy loss for known classes.
        '''
        known_logits = self.net.forward(batch, mask=known_mask, msg = 'labeled') # single layer labeled head 
        _, label_pred = torch.max(known_logits, dim = -1)
        known_label = labels[known_mask]
        acc = 1.0 * torch.sum(label_pred == known_label) / len(label_pred)
        ce_loss = F.cross_entropy(input = known_logits, target = known_label)
        return ce_loss, acc 


    def training_step(self, batch: List[Dict[str, torch.Tensor]], batch_idx:int):
        '''
        batch =  {
            'meta':List[Dict],
            'token_ids': torch.LongTensor (batch, seq_len),
            'attn_mask': torch.BoolTensor (batch, seq_len)
            'labels': torch.LongTensor([x['label'] for x in batch])
            'head_spans': ,
            'tail_spans': ,
            'mask_bpe_idx': ,
            'known_mask' torch.BoolTensor 
        }

        For RoCORE:  data = (input_ids, input_mask, label, head_span, tail_span)
        '''
        view_n = len(batch) 
        batch_size = len(batch[0]['meta'])
        labels = batch[0]['labels']
        known_mask = batch[0]['known_mask'] # (batch, )
        psuedo_labels = batch[0]['pseudo_labels'] 

        loss = self._compute_reconstruction_loss(batch[0], known_mask, labels)
        margin_loss = self._compute_unknown_margin_loss(batch[0], psuedo_labels[~known_mask], known_mask)
        ce_loss, acc  =  self._compute_ce_loss(batch[0], known_mask, labels)
        if self.current_epoch >= self.config.num_pretrain_epochs:
            loss += margin_loss 
            loss += ce_loss 

            self.log('train/unknown_margin_loss', margin_loss)
            self.log('train/known_ce_loss', ce_loss)


        self.log('train/known_acc', acc)
        self.log('train/loss', loss)
        return loss 

    def validation_step(self, batch: List[Dict[str, torch.Tensor]], batch_idx:int, dataloader_idx:int):

        view_n = len(batch)
        batch_size = len(batch[0]['meta'])

       
          # use the gt labels to compute metrics 
        if dataloader_idx == 0:
            logits = self.net.forward(batch[0],msg = 'unlabeled')
            self.train_unknown_metrics_wrapper.update_batch(logits, batch[0]['labels']- self.config.known_types , incremental=False)
        elif dataloader_idx == 1:
            logits = self.net.forward(batch[0], msg = 'unlabeled')
            self.val_unknown_metrics_wrapper.update_batch(logits, batch[0]['labels'] - self.config.known_types, incremental=False)
        else:
            logits = self.net.forward(batch[0], msg = 'labeled')
            self.val_known_metrics_wrapper.update_batch(logits, batch[0]['labels'], incremental=False)
        return 

    def validation_epoch_end(self, outputs: List[List[Dict]]) -> None:
        val_unknown_metrics = self.val_unknown_metrics_wrapper.on_epoch_end() 
        train_unknown_metrics = self.train_unknown_metrics_wrapper.on_epoch_end() 
        val_known_metrics = self.val_known_metrics_wrapper.on_epoch_end() 

        for k,v in val_unknown_metrics.items():
            self.log(f'val/unknown_{k}',value=v, logger=True, on_step=False, on_epoch=True)

        for k,v in train_unknown_metrics.items():
            self.log(f'train/unknown_{k}', value=v, logger=True, on_step=False, on_epoch=True)

        for k,v in val_known_metrics.items():
            self.log(f'val/known_{k}',value=v, logger=True, on_step=False, on_epoch=True) 
        return 

    
    def test_step(self, batch: List[Dict], batch_idx: int) -> Dict:
        '''
        :param dataloader_idx: 0 for unknown_test
        '''
        view_n = len(batch)
        batch_size = len(batch[0]['meta'])

        target_logits = self.net.forward(batch[0], msg = 'unlabeled')
        # use the gt labels to compute metrics 
        self.test_unknown_metrics_wrapper.update_batch(target_logits, batch[0]['labels'] - self.config.known_types , incremental=False)
        
        self.predictions_wrapper.update_batch(batch[0]['meta'], target_logits, batch[0]['labels'] - self.config.known_types , incremental=False)

        return {} 

    def test_epoch_end(self, outputs: List[Dict]) -> None:
        test_unknown_metrics = self.test_unknown_metrics_wrapper.on_epoch_end() 
        for k,v in test_unknown_metrics.items():
            self.log(f'test/unknown_{k}',value=v, logger=True, on_step=False, on_epoch=True)
        
        self.test_unknown_metrics_wrapper.save(self.config.ckpt_dir)

        self.predictions_wrapper.on_epoch_end() 
        self.predictions_wrapper.save(self.config.ckpt_dir)

        return 


    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        return [optimizer, ]





        
