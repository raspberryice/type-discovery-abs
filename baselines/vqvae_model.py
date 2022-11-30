import os 
import json 
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm 

from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup

from common.metrics import ClusteringMetricsWrapper
from common.predictions import ClusterPredictionsWrapper
from common.utils import cluster_acc, get_label_mapping, onedim_gather
from baselines.vae import EventVQVAE



import common.log as log
logger = log.get_logger('root')


class VQVAEModel(pl.LightningModule):
    def __init__(self, args, tokenizer, train_len:int = 1000) -> None:
        super().__init__()
        self.config = args  
        
        self.tokenizer = tokenizer 
        self.model_config = AutoConfig.from_pretrained(args.model_name_or_path, output_hidden_states = True)
        self.pretrained_model = AutoModel.from_pretrained(args.model_name_or_path, config = self.model_config)
        embeddings = self.pretrained_model.resize_token_embeddings(len(self.tokenizer)) # when adding new tokens, the tokenizer.vocab_size is not changed! 

        self.train_len=train_len

        self.model = EventVQVAE(self.model_config.hidden_size, dim=500, 
            known_types=args.known_types, unknown_types=args.unknown_types, use_vae=True if args.hybrid else False)
        
        self.train_unknown_metrics_wrapper = ClusteringMetricsWrapper(stage='train',
            known=False,
            prefix='train_unknown', known_classes=args.known_types)
        self.val_unknown_metrics_wrapper = ClusteringMetricsWrapper(stage='val',
            known=False, 
            prefix='val_unknown',  known_classes=args.known_types)
        self.test_unknown_metrics_wrapper = ClusteringMetricsWrapper(stage='test',
            known=False,
            prefix='test_unknown',  known_classes=args.known_types)
        
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
            known_classes=args.known_types, task=args.task) 

    
    def on_validation_epoch_start(self) -> None:
        # reset all metrics
        self.train_unknown_metrics_wrapper.reset() 
        self.val_unknown_metrics_wrapper.reset() 
        self.test_unknown_metrics_wrapper.reset()

        self.train_known_metrics_wrapper.reset() 
        self.val_known_metrics_wrapper.reset() 
        self.test_known_metrics_wrapper.reset() 

        return

    def on_train_epoch_start(self) -> None:
        return 



    def training_step(self, batch: List[Dict[str, torch.Tensor]], batch_idx: int):
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
        '''
        view_n = len(batch) 
        batch_size = len(batch[0]['meta'])
        labels = batch[0]['labels']
        
        known_mask = batch[0]['known_mask'] # (batch, )
        view = batch[0] 
        outputs = self.pretrained_model(input_ids=view['token_ids'],attention_mask=view['attn_mask'])
        seq_output = outputs[0] 
        feat = onedim_gather(seq_output, dim=1, index=view['trigger_spans'][:, 0].unsqueeze(1)).squeeze(1)
        # (batch, hidden_dim)
        x_tilde, z_e_x, z_q_x, logits, kl_div = self.model(feat, known_mask, labels) 
        
        
        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, feat) + kl_div 
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())


        # supervised loss 
        loss_supervised = F.cross_entropy(logits[known_mask, :], labels[known_mask])
        # unsupervised margin loss 
        y = torch.softmax(logits, dim=1)
        known_prob = y[~known_mask, :self.config.known_types]
        unknown_prob = y[~known_mask, self.config.known_types:]
        diff = torch.max(known_prob, dim=1)[0] - torch.max(unknown_prob, dim=1)[0]
        loss_unsupervised = torch.sum(torch.clamp(diff, min=0))/diff.size(0)

        loss = self.config.beta * (loss_vq + loss_commit) + loss_supervised + self.config.gamma* loss_unsupervised +\
             self.config.recon_loss * loss_recons
        self.log('train/recon_loss', loss_recons)
        self.log('train/vq_loss', loss_vq+loss_commit)
        self.log('train/ss_loss', loss_supervised + loss_unsupervised)
        self.log('train/supervised_loss', loss_supervised)
        self.log('train/unsupervised_loss', loss_unsupervised)
        
        return loss


    def validation_step(self,  batch: List[Dict[str, torch.Tensor]], batch_idx: int, dataloader_idx: int)-> Dict[str, torch.Tensor]:
        '''
        :param dataloader_idx: 0 for unknown_train, 1 for unknown_test, 2 for known_test
        '''
        view_n = len(batch)
        batch_size = len(batch[0]['meta'])
        view = batch[0] 
        known_mask = batch[0]['known_mask'] # (batch, )

        outputs = self.pretrained_model(input_ids=view['token_ids'],attention_mask=view['attn_mask'])
        seq_output = outputs[0] 
        feat = onedim_gather(seq_output, dim=1, index=view['trigger_spans'][:, 0].unsqueeze(1)).squeeze(1)
        # (batch, hidden_dim)
        logits, indexes = self.model.encode(feat) 
    
        # use the gt labels to compute metrics 
        # for this model, do not evaluation on known types 
        # setting incremental to True will not subtract the number of known types 
        if dataloader_idx == 0:
            self.train_unknown_metrics_wrapper.update_batch(logits, batch[0]['labels'], incremental=False)
        elif dataloader_idx == 1:
            self.val_unknown_metrics_wrapper.update_batch(logits, batch[0]['labels'], incremental=False)
        else:
            self.val_known_metrics_wrapper.update_batch(logits, batch[0]['labels'], incremental=False)
        return {} 
    
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
        view = batch[0] 
        known_mask = batch[0]['known_mask'] # (batch, )

        outputs = self.pretrained_model(input_ids=view['token_ids'],attention_mask=view['attn_mask'])
        seq_output = outputs[0] 
        feat = onedim_gather(seq_output, dim=1, index=view['trigger_spans'][:, 0].unsqueeze(1)).squeeze(1)
        logits, indexes = self.model.encode(feat) 
        # use the gt labels to compute metrics 
        self.test_unknown_metrics_wrapper.update_batch(logits, batch[0]['labels'], incremental=False)
        
        self.predictions_wrapper.update_batch(batch[0]['meta'], logits, batch[0]['labels'], incremental=False)

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
    
        if self.config.max_steps > 0:
            t_total = self.config.max_steps
            self.config.num_train_epochs = self.config.max_steps // self.train_len // self.config.accumulate_grad_batches + 1
        else:
            t_total = self.train_len // self.config.accumulate_grad_batches * self.config.num_train_epochs

        logger.info('{} training steps in total.. '.format(t_total)) 
        
        
        # scheduler is called only once per epoch by default 
        scheduler =  get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=t_total)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'linear-schedule',
        }

        return [optimizer, ], [scheduler_dict,]

