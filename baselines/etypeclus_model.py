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

from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup


from .latent_space_clustering import AutoEncoder, cosine_dist
from common.metrics import ClusteringMetricsWrapper
from common.predictions import ClusterPredictionsWrapper
from common.utils import cluster_acc, get_label_mapping, onedim_gather



import common.log as log
logger = log.get_logger('root')

class ETypeClusModel(pl.LightningModule):
    '''
    This is a wrapper of the TopicCluster class.
    '''
    def __init__(self, args, tokenizer, train_len) -> None:
        super().__init__()
        self.config = args 
        self.tokenizer = tokenizer 
        self.model_config = AutoConfig.from_pretrained(args.model_name_or_path, output_hidden_states = True)
        self.pretrained_model = AutoModel.from_pretrained(args.model_name_or_path, config = self.model_config)
        embeddings = self.pretrained_model.resize_token_embeddings(len(self.tokenizer)) # when adding new tokens, the tokenizer.vocab_size is not changed! 
        
        self.train_len=train_len # this is required to set up the optimizer,make optional 
        self.temperature = args.temperature
        self.distribution = args.distribution

        input_dim = self.model_config.hidden_size 
        hidden_dims = eval(args.hidden_dims)
        self.topic_emb = nn.Parameter(torch.Tensor(args.unknown_types, hidden_dims[-1]))


        self.q_dict = {} # uid -> target distribution 

        self.model = AutoEncoder(input_dim, hidden_dims)
        
        
        torch.nn.init.xavier_normal_(self.topic_emb.data)

        self.freeze_model() 


        self.train_unknown_metrics_wrapper = ClusteringMetricsWrapper(stage='train',
            known=False,
            prefix='train_unknown', known_classes=args.known_types)
        self.val_unknown_metrics_wrapper = ClusteringMetricsWrapper(stage='val',
            known=False, 
            prefix='val_unknown',  known_classes=args.known_types)
        self.test_unknown_metrics_wrapper = ClusteringMetricsWrapper(stage='test',
            known=False,
            prefix='test_unknown',  known_classes=args.known_types)
        

        if args.eval_only:
            self.predictions_wrapper = ClusterPredictionsWrapper(reassign=True, prefix='test_unknown', 
            known_classes=args.known_types, task=args.task) 


    def on_validation_epoch_start(self) -> None:
        # reset all metrics
        self.train_unknown_metrics_wrapper.reset() 
        self.val_unknown_metrics_wrapper.reset() 
        self.test_unknown_metrics_wrapper.reset()
        return 


    def freeze_model(self):
        self.pretrained_model.requires_grad_(False)
        return 

    def on_train_epoch_start(self) -> None:
        train_dl = self.trainer.train_dataloader.loaders
        with torch.no_grad():
            z_list = []
            uid_list = []

            for batch in tqdm(iter(train_dl)):
                known_mask = batch[0]['known_mask']
                metadata = batch[0]['meta']
                batch_size = len(metadata) 

                view = batch[0]
                # move batch to gpu 
                for key in ['token_ids', 'attn_mask','head_spans','tail_spans','mask_bpe_idx','trigger_spans']:
                    if key in view: view[key] = view[key].to(self.device)
                # get features by pretrained_model 
                outputs = self.pretrained_model(input_ids=view['token_ids'],attention_mask=view['attn_mask'])
                seq_output = outputs[0] 
                feat = onedim_gather(seq_output, dim=1, index=view['trigger_spans'][:, 0].unsqueeze(1)).squeeze(1)
                x = F.normalize(feat[~known_mask], dim=1) 
                x_bar, z = self.model(x) 
                z = F.normalize(z, dim=1)
                z_list.append(z)
                uid_list.extend([it['uid'] for idx, it in enumerate(metadata) if known_mask[idx] == False]) 

        if self.current_epoch ==0:
            # initialize the clusters by kmeans 
            logger.info('initializing by kmeans...')
            kmeans = KMeans(n_clusters=self.config.unknown_types, n_init=5)
            rep = torch.concat(z_list, dim=0).cpu().numpy() 
            y_pred = kmeans.fit_predict(rep)  # y_pred is used to determine end of training 
            self.topic_emb.data = torch.tensor(kmeans.cluster_centers_).to(self.device)

        else:
            logger.info('updating target distribution q')
            all_z = torch.concat(z_list, dim=0) # (N, hidden_dim)
            freq = torch.ones((all_z.size(0)), dtype=torch.long, device=self.device) 
            p , q = self.target_distribution(all_z, freq, method='all', top_num=self.current_epoch+1)
            assert (q.size(0) == len(uid_list)) 
            for idx, uid in enumerate(uid_list):
                self.q_dict[uid] = q[idx,:]

        return 
    
    def collect_features(self, dataloader, known:bool=False, max_batch=50):
        '''
        collect features for visualization.
        '''
        feat_arrays = []
        
        with torch.no_grad():
            all_feats = [[],]
            all_labels = [[],] # List[str]
            
            for batch_idx, batch in enumerate(iter(dataloader)):
                view_n = len(batch) 
                known_mask = batch[0]['known_mask'] # (batch, )
                view = batch[0] 
                outputs = self.pretrained_model(input_ids=view['token_ids'],attention_mask=view['attn_mask'])
                seq_output = outputs[0] 
                feat = onedim_gather(seq_output, dim=1, index=view['trigger_spans'][:, 0].unsqueeze(1)).squeeze(1)
                if known:
                    feat = feat[known_mask]
                else:
                    feat = feat[~known_mask] 
                labels = [x['label'] for x in batch[0]['meta'] if x['known'] == known] 
                all_feats[0].append(feat.cpu().numpy())
                all_labels[0].extend(labels) 
                if batch_idx == max_batch: break 
            
            for i in range(len(all_feats)):
                all_feats_array = np.concatenate(all_feats[i], axis=0)# (n_instances, hidden_dim)
                feat_arrays.append(all_feats_array)
        
        with open(os.path.join(self.config.ckpt_dir, 'view_features.pkl'), 'wb') as f:
            pkl.dump(feat_arrays,f )
        with open(os.path.join(self.config.ckpt_dir, 'labels.pkl'),'wb') as f:
            pkl.dump(all_labels,f )

        return 
    def cluster_assign(self, z: torch.FloatTensor) -> torch.FloatTensor:
        '''
        :param z: (batch, hidden_dim)
        :returns p: (batch, n_clusters) 
        '''
        if self.distribution == 'student':
            p = 1.0 / (1.0 + torch.sum(
                torch.pow(z.unsqueeze(1) - self.topic_emb, 2), 2) / self.alpha)
            p = p.pow((self.alpha + 1.0) / 2.0)
            p = (p.t() / torch.sum(p, 1)).t()
        else:
            self.topic_emb.data = F.normalize(self.topic_emb.data, dim=-1)
            z = F.normalize(z, dim=-1)
            sim = torch.matmul(z, self.topic_emb.t()) / self.temperature
            p = F.softmax(sim, dim=-1)
        return p
    
    def forward(self, x: torch.FloatTensor):
        x_bar, z = self.model(x)
        p = self.cluster_assign(z)
        return x_bar, z, p

    def target_distribution(self, z: torch.FloatTensor, 
         freq: torch.LongTensor, method='all', top_num=0):
        '''
        :param x: (batch, hidden_dim)
        :param freq: (batch)
        ''' 
        p = self.cluster_assign(z).detach()
        if method == 'all':
            q = p**2 / (p * freq.unsqueeze(-1)).sum(dim=0)
            q = (q.t() / q.sum(dim=1)).t()
        elif method == 'top':
            assert top_num > 0
            q = p.clone()
            sim = torch.matmul(self.topic_emb, z.t())
            _, selected_idx = sim.topk(k=top_num, dim=-1)
            for i, topic_idx in enumerate(selected_idx):
                q[topic_idx] = 0
                q[topic_idx, i] = 1
        return p, q


    def training_step(self, batch, batch_idx):
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
        x = F.normalize(feat[~known_mask]) 

        x_bar, z , p = self.forward(x)
        reconstr_loss = cosine_dist(x_bar, x) 
        self.log('train/recon_loss', reconstr_loss)

        if self.current_epoch < self.config.num_pretrain_epochs: 
            loss = reconstr_loss
            return loss
        
        q_batch = torch.zeros((batch_size, self.config.unknown_types), dtype=torch.float, device=self.device)
        for i in range(batch_size):
            uid = batch[0]['meta'][i]['uid']
            if known_mask[i] == False:
                q_batch[i, : ] = self.q_dict[uid] 
        kl_loss = F.kl_div(p.log(), q_batch[~known_mask], reduction='none').sum()
        loss = self.config.gamma * kl_loss + reconstr_loss
        self.log('train/kl_loss',kl_loss)
      
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
        x = F.normalize(feat[~known_mask]) 

        x_bar, _ , p = self.forward(x)

        # use the gt labels to compute metrics 
        # for this model, do not evaluation on known types 
        # setting incremental to True will not subtract the number of known types 
        if dataloader_idx == 0:
            self.train_unknown_metrics_wrapper.update_batch(p, batch[0]['labels'], incremental=True)
        elif dataloader_idx == 1:
            self.val_unknown_metrics_wrapper.update_batch(p, batch[0]['labels'], incremental=True)
        
        return {} 
    

    def validation_epoch_end(self, outputs: List[List[Dict]]) -> None:
        val_unknown_metrics = self.val_unknown_metrics_wrapper.on_epoch_end() 
        train_unknown_metrics = self.train_unknown_metrics_wrapper.on_epoch_end() 

        for k,v in val_unknown_metrics.items():
            self.log(f'val/unknown_{k}',value=v, logger=True, on_step=False, on_epoch=True)

        for k,v in train_unknown_metrics.items():
            self.log(f'train/unknown_{k}', value=v, logger=True, on_step=False, on_epoch=True)

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
        x = F.normalize(feat[~known_mask]) 

        x_bar, _ , p = self.forward(x)
        
        # use the gt labels to compute metrics 
        self.test_unknown_metrics_wrapper.update_batch(p, batch[0]['labels'], incremental=True)
        
        self.predictions_wrapper.update_batch(batch[0]['meta'], p, batch[0]['labels'], incremental=True)

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




