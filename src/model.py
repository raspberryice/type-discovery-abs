import os 
import json
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm 

from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AdamW, get_linear_schedule_with_warmup

from common.sinkhorn_knopp import SinkhornKnopp
from multiview_layers import MultiviewModel
from common.metrics import ClusteringMetricsWrapper, PsuedoLabelMetricWrapper
from common.predictions import ClusterPredictionsWrapper
from common.utils import cluster_acc, get_label_mapping
from common.clustering import spectral_clustering, agglomerative_clustering, agglomerative_ward, dbscan


import common.log as log
logger = log.get_logger('root')

# torch.autograd.set_detect_anomaly(True) # for debugging 
UNK_LABEL=33 # TODO this is the label idx for instances that are not labeled 

from functools import wraps
from time import time

def timing(f):
    '''
    A decorator to time functions.
    '''
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.info(f'func:{f.__name__} took: {te-ts:2.4f} sec')
        return result
    return wrap


class TypeDiscoveryModel(pl.LightningModule):
    def __init__(self, args, tokenizer, train_len:int = 1000) -> None:
        super().__init__()
        self.config = args  
        
        self.tokenizer = tokenizer 
        self.model_config = AutoConfig.from_pretrained(args.model_name_or_path, output_hidden_states = True)
        if args.predict_names: # need to output names 
            pretrained_model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, config=self.model_config)
        else:
            pretrained_model = AutoModel.from_pretrained(args.model_name_or_path, config = self.model_config)
        embeddings = pretrained_model.resize_token_embeddings(len(self.tokenizer)) # when adding new tokens, the tokenizer.vocab_size is not changed! 

        self.train_len=train_len # this is required to set up the optimizer
        self.mv_model = MultiviewModel(args, self.model_config, pretrained_model, unfreeze_layers= [args.layer])
        # regularization
        self.sk = SinkhornKnopp(num_iters=3, epsilon=self.config.sk_epsilon, classes_n= args.unknown_types, queue_len=1024, delta=1e-10)

        # metrics 
        self.pl_metrics_wrapper = PsuedoLabelMetricWrapper(cache_size=512, 
            known_classes=args.known_types, 
            unknown_classes=args.unknown_types)

        self.view1_metrics_wrapper = PsuedoLabelMetricWrapper(cache_size=512, 
            known_classes=args.known_types, 
            unknown_classes=args.unknown_types)

        self.view2_metrics_wrapper = PsuedoLabelMetricWrapper(cache_size=512, 
            known_classes=args.known_types, 
            unknown_classes=args.unknown_types)

        self.train_unknown_metrics_wrapper = ClusteringMetricsWrapper(stage='train',
            known=False,
            prefix='train_unknown', known_classes=args.known_types)
        self.val_unknown_metrics_wrapper = ClusteringMetricsWrapper(stage='val',
            known=False, 
            prefix='val_unknown',  known_classes=args.known_types)
        self.test_unknown_metrics_wrapper = ClusteringMetricsWrapper(stage='test',
            known=False,
            prefix='test' if self.config.incremental else 'test_unknown',  known_classes=args.known_types)
        
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
            self.predictions_wrapper = ClusterPredictionsWrapper(reassign=True, prefix='test' if self.config.incremental else 'test_unknown', 
            known_classes=args.known_types, task=args.task, save_names=args.predict_names) 

    def on_validation_epoch_start(self) -> None:
        # reset all metrics
        self.train_unknown_metrics_wrapper.reset() 
        self.val_unknown_metrics_wrapper.reset() 
        self.test_unknown_metrics_wrapper.reset()

        self.train_known_metrics_wrapper.reset() 
        self.val_known_metrics_wrapper.reset() 
        self.test_known_metrics_wrapper.reset() 

        return 

    
    def load_pretrained_model(self, pretrained_dict):
        '''
        load model and common space proj,
        '''
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if ('pretrained_model' in k or 'common_space_proj' in k)}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
        return 

 
    @timing 
    def on_train_epoch_start(self) -> None:
        if self.config.supervised_pretrain: return 

        if self.current_epoch == 0 or self.config.clustering!='online':
            logger.info('updating cluster centers....')
            train_dl = self.trainer.train_dataloader.loaders
            known_centers = torch.zeros((2, self.config.known_types, self.config.kmeans_dim), device = self.device)

            num_samples = torch.zeros((2, self.config.known_types), device=self.device) 
            with torch.no_grad():
                uid2pl = defaultdict(list) # pseudo labels 
                unknown_uid_list = defaultdict(list)
                unknown_vec_list = defaultdict(list)
                seen_uid = [set(), set()] # oversampling for the unknown part, so we remove them here 
                for batch in tqdm(iter(train_dl)):
                    labels = batch[0]['labels']
                    known_mask = batch[0]['known_mask']
                    metadata = batch[0]['meta']
                    batch_size = len(metadata) 

                    for view_idx, view in enumerate(batch):
                        # move batch to gpu 
                        for key in ['token_ids', 'attn_mask','head_spans','tail_spans','mask_bpe_idx','trigger_spans']:
                            if key in view: view[key] = view[key].to(self.device)
                        
                        feature_type = view['meta'][0]['feature_type']
                        feats = self.mv_model._compute_features(view, feature_type, pooling=self.config.token_pooling)
                        view_model = self.mv_model.views[view_idx]
                        commonspace_rep = view_model['common_space_proj'](feats) # (batch_size, hidden_dim)
                        for i in range(batch_size):
                            if known_mask[i] == True:
                                l = labels[i]
                                known_centers[view_idx][l] += commonspace_rep[i]
                                num_samples[view_idx][l] += 1 
                            else:
                                uid = metadata[i]['uid']
                                if uid not in seen_uid[view_idx]:
                                    seen_uid[view_idx].add(uid)
                                    unknown_uid_list[view_idx].append(uid)
                                    unknown_vec_list[view_idx].append(commonspace_rep[i])
                
                assert len(unknown_uid_list[0]) == len(unknown_uid_list[1]) 
                for view_idx in range(2):
                    # cluster unknown classes
                    rep = torch.stack(unknown_vec_list[view_idx], dim=0).cpu().numpy()  
                    if self.config.clustering == 'spectral':
                        label_pred = spectral_clustering(rep, self.config.unknown_types)
                    elif self.config.clustering == 'agglomerative':
                        label_pred = agglomerative_clustering(rep, self.config.unknown_types)
                    elif self.config.clustering == 'ward': 
                        label_pred = agglomerative_ward(rep, self.config.unknown_types) 
                    elif self.config.clustering == 'dbscan':
                        # TODO: eps hyperparameter is very sensitive and only set for TACRED here 
                        label_pred, eps  = dbscan(rep, self.config.unknown_types, eps=None if self.current_epoch==0 else self.eps)
                        self.eps = eps 
                        logger.info(f'found {np.max(label_pred)} clusters using DBScan with eps {eps}') 
                    else: # default is kmeans 
                        clf = KMeans(n_clusters=self.config.unknown_types,random_state=0,algorithm='full')
                        label_pred = clf.fit_predict(rep)# from 0 to args.new_class - 1
                        
                    for i in range(len(unknown_vec_list[view_idx])): 
                        uid = unknown_uid_list[view_idx][i]
                        pseudo = label_pred[i]
                        uid2pl[uid].append( pseudo + self.config.known_types)

                    # update center for known types 
                    for c in range(self.config.known_types):
                        known_centers[view_idx][c] /= num_samples[view_idx][c]
            
                train_dl.dataset.update_pseudo_labels(uid2pl) 
                logger.info('updating pseudo labels...')
                pl_acc = train_dl.dataset.check_pl_acc() 
                self.log('train/kmeans_acc', pl_acc, on_epoch=True)
                
        return


    def _compute_psuedo_labels(self, pred_logits: torch.FloatTensor, pred_logits_other: torch.FloatTensor, 
        known_mask: torch.BoolTensor, targets: torch.FloatTensor, 
        temp:float=0.5, regularization: str='sk', mode: str='other') -> Union[torch.FloatTensor, None]:
        '''
        Compute the psuedo labels by combine and sharpen.
        :param pred_logits: (batch, M+N)
        :param pred_logits_other: (batch, M+N)
        :param known_mask: (batch)
        :param targets: (batch, M+N)
        :param temp: float in (0, 1)
        :param regularization: str, 
        :param mode: str, one of 'other', 'combine'
        '''
        assert mode in ['self','other','combine'], f"invalid mode {mode}"
        if mode == 'self':
            target_logits = pred_logits 
        elif mode == 'other':
            target_logits = pred_logits_other
        elif mode == 'combine': 
            target_logits = pred_logits + pred_logits_other
        
        target_logits_detached = target_logits.detach()

        if self.config.rev_ratio > 0: 
            known_types = self.config.known_types *2 
        else:
            known_types = self.config.known_types

        if regularization == 'sk' :
            # solve sinkhorn knopp
            input_logits = target_logits_detached[~known_mask, known_types:]
            self.sk.add_to_queue(input_logits)
            if self.sk.queue_full:
                prob, row_sum, col_sum = self.sk(input_logits) 
                assert torch.any(torch.isnan(prob)) == False, "sk result contains nan" 
                targets[~known_mask, known_types:] = prob
            
            else:
                return None 

        else:
            targets[~known_mask, known_types:] = torch.softmax(target_logits_detached[~known_mask, known_types:]/temp, dim=1)
        
        return targets 


    def _smooth_targets(self, targets:torch.FloatTensor, all_types: int):
        if self.config.label_smoothing_alpha > 0.0:
            if self.config.label_smoothing_ramp > 0:
                alpha =( 1- self.current_epoch*1.0/self.config.label_smoothing_ramp) * self.config.label_smoothing_alpha
            else:
                alpha = self.config.label_smoothing_alpha

            alpha = max(alpha, 0.0)
            targets = (1-alpha) * targets + alpha * torch.full_like(targets, fill_value=1.0/all_types, dtype=torch.float, device=self.device)
        
        return targets 

    def _compute_targets(self, batch_size:int, labels: torch.LongTensor, known_mask: torch.BoolTensor, 
            predicted_logits:torch.FloatTensor, predicted_logits_other: torch.FloatTensor, hard: bool=False):
        if self.config.rev_ratio > 0: 
            targets = torch.zeros((batch_size, 2* self.config.known_types+  self.config.unknown_types), dtype=torch.float, device=self.device) # soft targets 
            all_types= 2* self.config.known_types+  self.config.unknown_types
        else:
            targets = torch.zeros((batch_size, self.config.known_types+self.config.unknown_types), dtype=torch.float, device=self.device) # soft targets 
            all_types= self.config.known_types+self.config.unknown_types
        
        assert (labels.max() < all_types) 
        known_labels = F.one_hot(labels, num_classes=all_types).float() # (batch, all_types)

        
        assert (known_mask.long().max() <= 1 and known_mask.long().min() >=0)
        if known_mask.long().max() > 0: # has known elements  
            targets[known_mask, :] = known_labels[known_mask, :]

        if known_mask.long().min() < 1: # has unknown elements 
            # compute psuedo labels 
            targets = self._compute_psuedo_labels(predicted_logits, predicted_logits_other, known_mask, targets, 
                temp=self.config.temp, regularization=self.config.regularization,
                mode=self.config.psuedo_label)

            targets_other = self._compute_psuedo_labels(predicted_logits_other, predicted_logits, known_mask, targets, 
                temp=self.config.temp, regularization=self.config.regularization,
                mode=self.config.psuedo_label)
        else:
            targets_other = targets 
        

        targets = self._smooth_targets(targets, all_types)
        targets_other = self._smooth_targets(targets_other, all_types)

        return targets, targets_other 



    def collect_features(self, dataloader, known:bool=False, raw: bool=False, max_batch=100):
        '''
        collect features for visualization.
        :param raw: when set to true, features before the projection layer 
        '''
        feat_arrays = []
        
        with torch.no_grad():
            all_feats = [[],[]]
            all_labels = [[],[]] # List[str]
            
            for batch_idx, batch in enumerate(iter(dataloader)):
                view_n = len(batch) 
                known_mask = batch[0]['known_mask'] # (batch, )
                for i in range(view_n):
                    feature_type = batch[i]['meta'][0]['feature_type']
                    feats = self.mv_model._compute_features(batch[i], feature_type, pooling=self.config.token_pooling)
                    if not raw:
                        view_model = self.mv_model.views[i]
                        if known: 
                            common_space_feats = view_model['common_space_proj'](feats[known_mask]) # (unknown_n, hidden_dim)
                        else:
                            common_space_feats = view_model['common_space_proj'](feats[~known_mask]) # (unknown_n, hidden_dim)
                        # common_space_feats = F.normalize(common_space_feats,dim =1, p=2)
                        all_feats[i].append(common_space_feats.cpu().numpy())
                    else:
                        if known: feats = feats[known_mask]
                        else: feats = feats[~known_mask] 
                        
                        all_feats[i].append(feats.cpu().numpy())
                    labels = [x['label'] for x in batch[0]['meta'] if x['known'] == known] 
                    all_labels[i].extend(labels) 

                if batch_idx == max_batch: break 
            
            for i in range(len(all_feats)):
                all_feats_array = np.concatenate(all_feats[i], axis=0)# (n_instances, hidden_dim)
                feat_arrays.append(all_feats_array)
        
        with open(os.path.join(self.config.ckpt_dir, 'view_features.pkl'), 'wb') as f:
            pkl.dump(feat_arrays,f )
        with open(os.path.join(self.config.ckpt_dir, 'labels.pkl'),'wb') as f:
            pkl.dump(all_labels,f )

        return 


    def _compute_batch_pairwise_loss(self,
        predicted_logits: torch.FloatTensor, 
        labels: Optional[torch.LongTensor]=None,
        targets: Optional[torch.FloatTensor]=None,
        loss_fn: str='kl', sigmoid:float =2.0):
        '''
        targets: (batch, M+N), probabilities 
        predicted_logits: (batch, M+N)
        known_mask: (batch)
        loss_fn: one of 'bce', 'kl', 'l2' 
        '''   

        if self.config.rev_ratio > 0: 
            known_types = self.config.known_types *2 
        else:
            known_types = self.config.known_types

        predicted_logits = predicted_logits[:, known_types:] 

        def compute_kld(p_logit, q_logit):
            p = F.softmax(p_logit, dim = -1) # (B, B, n_class) 
            q = F.softmax(q_logit, dim = -1) # (B, B, n_class)
            return torch.sum(p * (torch.log(p + 1e-16) - torch.log(q + 1e-16)), dim = -1) # (B, B)
        
        if targets != None:
            targets = targets[:, known_types:]
            assert (targets.shape == predicted_logits.shape)
            # convert targets into pairwise labels 
            targets = targets.detach() 
            batch_size = targets.size(0)
            target_val, target_idx = torch.max(targets, dim=1)
            pairwise_label = (target_idx.unsqueeze(0) == target_idx.unsqueeze(1)).float() # (batch, batch)
        else:
            batch_size = labels.size(0)
            label_mask = (labels != -1)
            
            pairwise_label = (labels[label_mask].unsqueeze(0) == labels[label_mask].unsqueeze(1)).float()
            predicted_logits = predicted_logits[label_mask]

        if loss_fn == 'kl':
            expanded_logits = predicted_logits.expand(batch_size, -1, -1)
            expanded_logits2 = expanded_logits.transpose(0, 1)
            kl1 = compute_kld(expanded_logits.detach(), expanded_logits2)
            kl2 = compute_kld(expanded_logits2.detach(), expanded_logits) # (batch_size, batch_size)
            pair_loss = torch.mean(pairwise_label * (kl1 + kl2) + (1 - pairwise_label) * (torch.relu(sigmoid - kl1) + torch.relu(sigmoid - kl2)))
        elif loss_fn== 'bce':
            pij = F.cosine_similarity(predicted_logits.unsqueeze(0), predicted_logits.unsqueeze(1), dim=2)
            pair_loss = F.binary_cross_entropy_with_logits(pij, pairwise_label)
        
        else:    
            raise NotImplementedError

        return pair_loss 

    
    def _compute_consistency_loss(self, predicted_logits: torch.FloatTensor, 
            predicted_logits_other: torch.FloatTensor, known_mask: torch.BoolTensor, 
            loss_fc='l2'):
        '''
        :param predicted_logits (N, n_classes)
        :param predicted_logits_other (N, n_classes)
        :param type: one of l2, kl
        '''
        assert (predicted_logits.shape == predicted_logits_other.shape)
        if loss_fc == 'l2':# average over number of classes and batch_size
            consistency_loss = torch.mean(torch.mean(torch.square(predicted_logits- predicted_logits_other), dim=1))
        
        elif loss_fc=='kl':
            def compute_kld(p_logit, q_logit):
                p = F.softmax(p_logit, dim = -1) # (B, n_class) 
                q = F.softmax(q_logit, dim = -1) # (B, n_class)
                return torch.sum(p * (torch.log(p + 1e-16) - torch.log(q + 1e-16)), dim = -1) # (B,)
            
            kl1 = compute_kld(predicted_logits.detach(), predicted_logits_other)
            kl2 = compute_kld(predicted_logits_other.detach(), predicted_logits)

            consistency_loss = torch.mean( kl1+kl2) * 0.5
            

        else:
            raise NotImplementedError
        return consistency_loss 



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
        self.mv_model._on_train_batch_start() 

        view_n = len(batch) 
        batch_size = len(batch[0]['meta'])
        labels = batch[0]['labels']
        
        known_mask = batch[0]['known_mask'] # (batch, )

        contrastive_loss = 0.0
        contrastive_loss_other = 0.0 
        pos_acc = 1.0
        recon_loss = 0.0
        recon_loss_other = 0.0
        
        for i in range(view_n):
            pl = batch[i]['pseudo_labels'] #(batch)
            # check feature type 
            feature_type = batch[i]['meta'][0]['feature_type']
            if i==0:
                predicted_logits, commonspace_feat, feat = self.mv_model._compute_prediction_logits( 
                    batch[i], method=feature_type, pooling=self.config.token_pooling, view_idx=i)
            else:
                predicted_logits_other, commonspace_feat_other, feat_other = self.mv_model._compute_prediction_logits(
                    batch[i], method=feature_type, pooling=self.config.token_pooling, view_idx=i)
          
        if self.config.supervised_training:
            targets = targets_other = F.one_hot(labels, num_classes=self.config.known_types+self.config.unknown_types).float()
        
        else: # targets are the same for the two views 
            targets, targets_other = self._compute_targets(batch_size, labels, known_mask, predicted_logits, predicted_logits_other, hard=False) 
            if (targets == None) or (targets_other== None): return 0.0 # no loss 

        known_loss = F.cross_entropy(predicted_logits[known_mask, :], target=targets[known_mask, :]) + F.cross_entropy(predicted_logits_other[known_mask,:], target=targets_other[known_mask, :])
        
        if self.config.supervised_pretrain:
            loss = known_loss 
        else:
            if self.config.pairwise_loss: 
                if self.config.clustering == 'kmeans':
                    pl_loss = self._compute_batch_pairwise_loss(predicted_logits[~known_mask],labels=batch[1]['pseudo_labels'][~known_mask], loss_fn='kl',sigmoid=self.config.sigmoid) \
                            + self._compute_batch_pairwise_loss(predicted_logits_other[~known_mask], labels=batch[0]['pseudo_labels'][~known_mask], loss_fn='kl', sigmoid=self.config.sigmoid)
                    self.log('train/unknown_margin_loss', pl_loss)
                else:
                    pl_loss = self._compute_batch_pairwise_loss(predicted_logits[~known_mask, :],targets=targets[~known_mask,:], loss_fn='kl') \
                        + self._compute_batch_pairwise_loss(predicted_logits_other[~known_mask, :], targets=targets_other[~known_mask, :], loss_fn='kl')
            else:
                assert self.config.clustering == 'online', 'unknown ce loss only works with online targets'
                if self.current_epoch == 0: # for the first epoch, use kmeans labels 
                    pl_loss = F.cross_entropy(predicted_logits[~known_mask], target=batch[0]['pseudo_labels'][~known_mask]) \
                        + F.cross_entropy(predicted_logits_other[~known_mask], target=batch[0]['pseudo_labels'][~known_mask])
                else:
                    pl_loss = F.cross_entropy(predicted_logits[~known_mask], target=targets[~known_mask, :]) \
                        + F.cross_entropy(predicted_logits_other[~known_mask], target=targets_other[~known_mask, :])

                
            
            loss =  known_loss + pl_loss 
            
        
        self.log('train/known_loss', known_loss)

        if self.config.contrastive_loss > 0:
            self.log('train/cl', contrastive_loss+ contrastive_loss_other)
            self.log('train/cl_acc', pos_acc) 
            loss += self.config.contrastive_loss * ( contrastive_loss + contrastive_loss_other) 
        
        if self.config.consistency_loss >0: 
            consistency_loss = self._compute_consistency_loss(predicted_logits, predicted_logits_other, known_mask, loss_fc='kl')
            self.log('train/consistency_loss', consistency_loss)
            loss += consistency_loss
        

        self.log('train/loss', loss)
       

        # evaluate psuedo label quality 
        if self.config.check_pl: 
            if known_mask.long().min() < 1: # has unknown elements 
                self.pl_metrics_wrapper.update_batch(targets[~known_mask, :], batch[0]['labels'][~known_mask])
                acc, ari = self.pl_metrics_wrapper.compute_metric()
                self.log('train/pl_acc', acc)

                self.view1_metrics_wrapper.update_batch(predicted_logits[~known_mask, :], batch[0]['labels'][~known_mask])
                view1_acc, _ = self.view1_metrics_wrapper.compute_metric() 
                self.log('train/view1_acc', view1_acc)

                self.view2_metrics_wrapper.update_batch(predicted_logits_other[~known_mask, :],batch[0]['labels'][~known_mask] )
                view2_acc, _ = self.view2_metrics_wrapper.compute_metric() 
                self.log('train/view2_acc', view2_acc)

    
        return loss

    
    def validation_step(self,  batch: List[Dict[str, torch.Tensor]], batch_idx: int, dataloader_idx: int)-> Dict[str, torch.Tensor]:
        '''
        :param dataloader_idx: 0 for unknown_train, 1 for unknown_test, 2 for known_test
        '''
        view_n = len(batch)
        batch_size = len(batch[0]['meta'])
        labels = batch[0]['labels']
        if self.config.e2e:
            annotated_mask = (labels!= UNK_LABEL)
        else:
            annotated_mask = (labels > -1) 
        for i in range(view_n):

            feature_type = batch[i]['meta'][0]['feature_type']
            if i==0:
                predicted_logits, _ , _ = self.mv_model._compute_prediction_logits( 
                    batch[i], method=feature_type, pooling=self.config.token_pooling, view_idx=0)
            else:
                predicted_logits_other, _ , _ = self.mv_model._compute_prediction_logits(
                    batch[i], method=feature_type, view_idx=1)

        
        if self.config.psuedo_label == 'combine':
            target_logits = predicted_logits + predicted_logits_other
        elif self.config.psuedo_label == 'self':
            target_logits = predicted_logits
        else:
            target_logits = predicted_logits_other 
        
        # use the gt labels to compute metrics 
        if dataloader_idx == 0:
            self.train_unknown_metrics_wrapper.update_batch(target_logits[annotated_mask], labels[annotated_mask], incremental=False)
        elif dataloader_idx == 1:
            self.val_unknown_metrics_wrapper.update_batch(target_logits[annotated_mask], labels[annotated_mask], incremental=False)
        else:
            self.val_known_metrics_wrapper.update_batch(target_logits, labels, incremental=False)
        
        
        
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
        labels = batch[0]['labels']
        if self.config.e2e:
            annotated_mask = (labels!= UNK_LABEL)
        else:
            annotated_mask = (labels > -1) 

        for i in range(view_n):
            feature_type = batch[i]['meta'][0]['feature_type']
            if i==0:
                predicted_logits, _, _  = self.mv_model._compute_prediction_logits( 
                    batch[i], method=feature_type, pooling=self.config.token_pooling,view_idx=0)
            else:
                predicted_logits_other, _, _ = self.mv_model._compute_prediction_logits(
                    batch[i], method=feature_type,view_idx=1)

            if feature_type == 'mask' and self.config.predict_names:
                predicted_token_ids = self.mv_model.predict_name(batch[i]) # (batch, topk)
                predicted_names = self.tokenizer.batch_decode(predicted_token_ids) # type: List[str]
            else:
                predicted_names = None 
            
        if self.config.psuedo_label == 'combine':
            target_logits = predicted_logits + predicted_logits_other
        elif self.config.psuedo_label == 'self':
            target_logits = predicted_logits
        else:
            target_logits = predicted_logits_other 
        

        # use the gt labels to compute metrics 
        self.test_unknown_metrics_wrapper.update_batch(target_logits[annotated_mask], labels[annotated_mask], incremental=self.config.incremental)
        
        self.predictions_wrapper.update_batch(batch[0]['meta'], target_logits, batch[0]['labels'], incremental=self.config.incremental, names=predicted_names)

        return {} 


    def test_epoch_end(self, outputs: List[Dict]) -> None:
        test_unknown_metrics = self.test_unknown_metrics_wrapper.on_epoch_end() 
        for k,v in test_unknown_metrics.items():
            prefix = 'test/' if self.config.incremental else 'test/unknown_'
            self.log(f'{prefix}{k}',value=v, logger=True, on_step=False, on_epoch=True)
        
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



