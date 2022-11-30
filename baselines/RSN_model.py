import os 
import json 
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import pickle as pkl
import random
from tqdm import tqdm 


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup
from torchmetrics import F1Score 

from common.metrics import PairwiseClusteringMetricsWrapper
from common.predictions import PairwiseClusterPredictionsWrapper
from .RSN import RSNLayer 

import common.log as log
logger = log.get_logger('root')

class RSNModel(pl.LightningModule):
    def __init__(self, args, tokenizer, train_len:int = 1000) -> None:
        super().__init__()
        self.config = args  
        self.tokenizer = tokenizer 
        self.model_config = AutoConfig.from_pretrained(args.model_name_or_path, output_hidden_states = True)
        self.pretrained_model = AutoModel.from_pretrained(args.model_name_or_path, config = self.model_config)
        embeddings = self.pretrained_model.resize_token_embeddings(len(self.tokenizer)) # when adding new tokens, the tokenizer.vocab_size is not changed! 

        self.train_len=train_len

        self.rsn_head = RSNLayer(self.model_config.hidden_size, rel_dim=self.config.hidden_dim, use_cnn=self.config.use_cnn) 

        self.f1_metric = F1Score(num_classes=1, threshold=0.5, multiclass=False)

        self.train_unknown_metrics_wrapper = PairwiseClusteringMetricsWrapper(stage='train',
            prefix='train_unknown', cluster_n=args.unknown_types, clustering_method=self.config.clustering_method)
        self.val_unknown_metrics_wrapper = PairwiseClusteringMetricsWrapper(stage='val',
            prefix='val_unknown',  cluster_n=args.unknown_types, clustering_method=self.config.clustering_method) 
        self.test_unknown_metrics_wrapper = PairwiseClusteringMetricsWrapper(stage='test',
            prefix='test_unknown',  cluster_n=args.unknown_types, clustering_method=self.config.clustering_method)
        
        self.val_known_metrics_wrapper = PairwiseClusteringMetricsWrapper(stage='val',
            prefix='val_known', cluster_n=args.known_types, clustering_method=self.config.clustering_method)

        if args.eval_only:
            self.predictions_wrapper = PairwiseClusterPredictionsWrapper(prefix='test_unknown', 
             task=args.task) 

    # def get_v_adv_loss(self, ul_left_input, ul_right_input, p_mult, power_iterations=1):
    #     bernoulli = tf.distributions.Bernoulli
    #     prob, left_word_emb, right_word_emb = self(ul_left_input, ul_right_input)[0:3]
    #     prob = tf.clip_by_value(prob, 1e-7, 1.-1e-7)
    #     prob_dist = bernoulli(probs=prob)
    #     #generate virtual adversarial perturbation
    #     left_d = tf.random_uniform(shape=tf.shape(left_word_emb), dtype=tf.float32)
    #     right_d = tf.random_uniform(shape=tf.shape(right_word_emb), dtype=tf.float32)
    #     for _ in range(power_iterations):
    #         left_d = (0.02) * tf.nn.l2_normalize(left_d, dim=1)
    #         right_d = (0.02) * tf.nn.l2_normalize(right_d, dim=1)
    #         p_prob = tf.clip_by_value(self(ul_left_input, ul_right_input, left_d, right_d)[0], 1e-7, 1.-1e-7)
    #         kl = tf.distributions.kl_divergence(prob_dist, bernoulli(probs=p_prob), allow_nan_stats=False)
    #         left_gradient,right_gradient = tf.gradients(kl, [left_d,right_d],
    #             aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    #         left_d = tf.stop_gradient(left_gradient)
    #         right_d = tf.stop_gradient(right_gradient)
    #     left_d = p_mult * tf.nn.l2_normalize(left_d, dim=1)
    #     right_d = p_mult * tf.nn.l2_normalize(right_d, dim=1)
    #     tf.stop_gradient(prob)
    #     #virtual adversarial loss
    #     p_prob = tf.clip_by_value(self(ul_left_input, ul_right_input, left_d, right_d)[0], 1e-7, 1.-1e-7)
    #     v_adv_losses = tf.distributions.kl_divergence(prob_dist, bernoulli(probs=p_prob), allow_nan_stats=False)
    #     return tf.reduce_mean(v_adv_losses)


    def compute_vat_loss(self, head_spans, tail_spans, seq_output, 
            perturb_scale:float=0.02, power_iterations: int=1):
        seq_output  = seq_output.detach() 
        prob = self.rsn_head(head_spans, tail_spans, seq_output) # B, B 
        prob = torch.clamp(prob, min=1e-7, max=1.0-1e-7)
        prob_dist = torch.distributions.Bernoulli(probs=prob)
        prob = prob.detach() 
        
        # generate perturbation
        d = torch.rand_like(seq_output, dtype=torch.float, requires_grad=True, device=self.device)
        for _ in range(power_iterations):
            d = perturb_scale * F.normalize(d, p=2, dim=2)
            p_prob = self.rsn_head(head_spans, tail_spans, seq_output, perturb=d)
            p_prob =  torch.clamp(p_prob, min=1e-7, max=1.0-1e-7)
            kl = torch.distributions.kl_divergence(prob_dist, torch.distributions.Bernoulli(probs=p_prob))
            kl = torch.mean(kl)
            kl.backward(inputs=d)

            d_grad = torch.clone(d.grad)
            d.grad.zero_()  
        d = perturb_scale * F.normalize(d_grad, p=2, dim=2)
        
        p_prob = self.rsn_head(head_spans, tail_spans, seq_output, perturb=d) 
        p_prob = torch.clamp(p_prob, min=1e-7, max=1.0-1e-7)
        vat_loss = torch.mean(torch.distributions.kl_divergence(prob_dist, torch.distributions.Bernoulli(probs=p_prob)))

        return vat_loss 


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
        
        head_spans = view['head_spans'] # (batch, 2)
        tail_spans = view['tail_spans'] # (batch,2 )

        pair_predicted_known = self.rsn_head(head_spans[known_mask], tail_spans[known_mask], seq_output[known_mask]) # (B, B)
        # 0 for same class and 1 for different class 
        pair_labels_known = (labels[known_mask].unsqueeze(1) != labels[known_mask].unsqueeze(0)).float() # (B, B)
        # supervised bce loss 
        bce_loss = F.binary_cross_entropy(pair_predicted_known, pair_labels_known)
        
        if known_mask.long().min() < 1: # has unknown elements 
            pair_predicted_unknown = self.rsn_head(head_spans[~known_mask], tail_spans[~known_mask],seq_output[~known_mask, :]) # (B, B)
            # unsupervised conditional entropy loss 
            pair_predicted_unknown = torch.clamp(pair_predicted_unknown, min=1e-7, max=1-1e-7)
            cond_loss = F.binary_cross_entropy(pair_predicted_unknown, pair_predicted_unknown) 
        else:
            cond_loss = 0.0
        
        if self.config.vat_loss_weight >0:
            sup_vat_loss = self.compute_vat_loss(head_spans[known_mask], tail_spans[known_mask], seq_output[known_mask], 
                perturb_scale=self.config.perturb_scale, power_iterations=1)
            unsup_vat_loss= self.compute_vat_loss(head_spans[~known_mask], tail_spans[~known_mask], seq_output[~known_mask], 
                perturb_scale=self.config.perturb_scale, power_iterations=1)
        else:
            sup_vat_loss = 0.0 
            unsup_vat_loss = 0.0 
        supervised_loss = bce_loss + self.config.vat_loss_weight * sup_vat_loss 
        unsupervised_loss = cond_loss + self.config.vat_loss_weight * unsup_vat_loss 
        loss = supervised_loss + self.config.p_cond * unsupervised_loss
        
        self.log('train/bce_loss', bce_loss)
        self.log('train/cond_ent_loss', cond_loss)
        self.log('train/loss', loss)


        return loss 
    


    def validation_step(self,  batch: List[Dict[str, torch.Tensor]], batch_idx: int, dataloader_idx: int)-> Dict[str, torch.Tensor]:
        '''
        :param dataloader_idx: 0 for unknown_train, 1 for unknown_test, 2 for known_test
        '''
        view_n = len(batch)
        batch_size = len(batch[0]['meta'])
        view = batch[0] 
        known_mask = batch[0]['known_mask'] # (batch, )
        labels = batch[0]['labels']
        
        outputs = self.pretrained_model(input_ids=view['token_ids'],attention_mask=view['attn_mask'])
        seq_output = outputs[0] 

        head_spans = view['head_spans'] # (batch, 2)
        tail_spans = view['tail_spans'] # (batch,2 )
        rel_embed = self.rsn_head.embed(head_spans, tail_spans, seq_output) # (B, rel_dim)
        return {
            'meta': batch[0]['meta'],
            'labels': batch[0]['labels'],
            'embed': rel_embed 
        } 

        # pair_predicted = self.rsn_head(head_spans, tail_spans, seq_output)
        # pair_labels = (labels.unsqueeze(1) != labels.unsqueeze(0)) # (B, B)

        # acc = torch.sum((pair_predicted > 0.5) == pair_labels)/ (batch_size * batch_size) 

        # if dataloader_idx == 0:
        #     self.log('train/unknown_pair_acc', acc, on_epoch=True, add_dataloader_idx=False) 
        # elif dataloader_idx ==1:
        #     self.log('val/unknown_pair_acc', acc, on_epoch=True, add_dataloader_idx=False)
        # else:
        # self.log('val/known_pair_acc', acc, on_epoch=True, add_dataloader_idx=False) 

        # return {}
    
    def validation_epoch_end(self, all_outputs: List[List[Dict]]) -> None:
        VAL_SAMPLE=1000
        for dataloader_idx, outputs in enumerate(all_outputs):
            # flatten list 
            if dataloader_idx != 1: continue # only use val_unknown 
            embeddings = {} # uid -> tensor 
            metadata = {} 
            for output in outputs:
                batch_size = len(output['meta'])
                for i in range(batch_size):
                    uid = output['meta'][i]['uid']
                    embeddings[uid] = output['embed'][i] 
                    metadata[uid] = output['meta'][i] 
                    metadata[uid]['label_idx'] = output['labels'][i].item() 
            # compute similarity 
            distance = {} 
            labels = {} 
            seen = set() 
            if len(embeddings) > VAL_SAMPLE:
                sampled_uids = random.sample(list(embeddings.keys()), k=VAL_SAMPLE) 
            else:
                sampled_uids = embeddings.keys() 
            
            
            for uid in sampled_uids:
                x1= embeddings[uid]
                labels[uid] = metadata[uid]['label_idx']
                for uid_other in sampled_uids: 
                    x2 = embeddings[uid_other]
                    if (uid, uid_other) not in seen:
                        dis = self.rsn_head.compute_distance(x1, x2).item()
                        distance[(uid, uid_other)] = dis
                        distance[(uid_other, uid)] = dis 
                        seen.add((uid, uid_other))
                        seen.add((uid_other, uid)) 

            sampled_labels = {k:labels[k] for k in sampled_uids}
            logger.info(f'running clustering on {len(labels)} data points...')
            val_unknown_metrics, _ = self.val_unknown_metrics_wrapper.on_epoch_end(distance, sampled_labels)

        for k,v in val_unknown_metrics.items():
            self.log(f'val/unknown_{k}',value=v, logger=True, on_step=False, on_epoch=True)

        return 


    def test_step(self, batch: List[Dict], batch_idx: int) -> Dict:
        '''
        :param dataloader_idx: 0 for unknown_test
        '''
        view_n = len(batch)
        batch_size = len(batch[0]['meta'])
        view = batch[0] 
        outputs = self.pretrained_model(input_ids=view['token_ids'],attention_mask=view['attn_mask'])
        seq_output = outputs[0] 
        head_spans = view['head_spans'] # (batch, 2)
        tail_spans = view['tail_spans'] # (batch,2 )
        rel_embed = self.rsn_head.embed(head_spans, tail_spans, seq_output) # (B, rel_dim)
        return {
            'meta': batch[0]['meta'],
            'labels': batch[0]['labels'],
            'embed': rel_embed 
        } 

        
    def test_epoch_end(self, outputs: List[Dict]) -> None:
        # flatten list 
        embeddings = {} # uid -> tensor 
        metadata = {} 
        for output in outputs:
            batch_size = len(output['meta'])
            for i in range(batch_size):
                uid = output['meta'][i]['uid']
                embeddings[uid] = output['embed'][i] 
                metadata[uid] = output['meta'][i] 
                metadata[uid]['label_idx'] = output['labels'][i].item() 
        # compute similarity 
        distance = {} 
        labels = {} 
        seen = set() 
        for uid in tqdm(embeddings):
            x1= embeddings[uid]
            labels[uid] = metadata[uid]['label_idx']
            for uid_other in embeddings: 
                x2 = embeddings[uid_other]
                if (uid, uid_other) not in seen:
                    dis = self.rsn_head.compute_distance(x1, x2).item()
                    distance[(uid, uid_other)] = dis
                    distance[(uid_other, uid)] = dis 
                    seen.add((uid, uid_other))
                    seen.add((uid_other, uid)) 

        logger.info(f'running clustering on {len(labels)} data points...')

        test_unknown_metrics, pred_cluster = self.test_unknown_metrics_wrapper.on_epoch_end(distance, labels) 
        for k,v in test_unknown_metrics.items():
            self.log(f'test/unknown_{k}',value=v, logger=True, on_step=False, on_epoch=True)
        
        self.test_unknown_metrics_wrapper.save(self.config.ckpt_dir)

        self.predictions_wrapper.on_epoch_end(pred_cluster, metadata) 
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



