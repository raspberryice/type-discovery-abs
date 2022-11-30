from typing import List, Dict, Optional 

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.center_loss import CenterLoss

'''
Code from RoCORE/model.py
'''

def L2Reg(net):
    reg_loss = 0
    for name, params in net.named_parameters():
        if name[-4:] != 'bias':
            reg_loss += torch.sum(torch.pow(params, 2))
    return reg_loss


def compute_kld(p_logit, q_logit):
    p = F.softmax(p_logit, dim = -1) # (B, B, n_class) 
    q = F.softmax(q_logit, dim = -1) # (B, B, n_class)
    return torch.sum(p * (torch.log(p + 1e-16) - torch.log(q + 1e-16)), dim = -1) # (B, B)
    


class ZeroShotModel(nn.Module):
    def __init__(self, args, known_types: int, unknown_types: int, model_config, pretrained_model, unfreeze_layers = []):
        super().__init__()
        # self.IL = args.IL
        self.known_types = known_types 
        self.unknown_types = unknown_types 
        self.hidden_dim = args.hidden_dim
        self.kmeans_dim = args.kmeans_dim
        self.initial_dim = model_config.hidden_size
        self.unfreeze_layers = unfreeze_layers
        self.pretrained_model = self.finetune(pretrained_model, self.unfreeze_layers) # fix bert weights
        self.layer = args.layer


        self.similarity_encoder = nn.Sequential(
                nn.Linear(2 * self.initial_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.kmeans_dim)
        )
        self.similarity_decoder = nn.Sequential(
                nn.Linear(self.kmeans_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, 2 * self.initial_dim)
        )
        self.ct_loss_u = CenterLoss(dim_hidden = self.kmeans_dim, num_classes = self.unknown_types, alpha=1.0, weight_by_prob=True)
        self.ct_loss_l = CenterLoss(dim_hidden = self.kmeans_dim, num_classes = self.known_types)
        # if self.IL:
        # self.labeled_head = nn.Linear(2 * self.initial_dim, self.known_types + self.unknown_types)
        self.labeled_head = nn.Linear(2 * self.initial_dim, self.known_types)
        self.unlabeled_head = nn.Linear(2 * self.initial_dim, self.unknown_types)
        self.bert_params = []
        for name, param in self.pretrained_model.named_parameters():
            if param.requires_grad is True:
                self.bert_params.append(param)

    @staticmethod
    def finetune(model, unfreeze_layers):
        params_name_mapping = ['embeddings', 'layer.0', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6', 'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11', 'layer.12']
        for name, param in model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if params_name_mapping[ele] in name:
                    param.requires_grad = True
                    break
        return model
 
    def get_pretrained_feature(self, input_id, input_mask, head_span, tail_span):
        outputs = self.pretrained_model(input_id, token_type_ids=None, attention_mask=input_mask) # (13 * [batch_size, seq_len, bert_embedding_len])
        all_encoder_layers = outputs[2]
        encoder_layers = all_encoder_layers[self.layer] # (batch_size, seq_len, bert_embedding)
        batch_size = encoder_layers.size(0)
        head_entity_rep = torch.stack([torch.max(encoder_layers[i, head_span[i][0]:head_span[i][1], :], dim = 0)[0] for i in range(batch_size)], dim = 0)
        tail_entity_rep = torch.stack([torch.max(encoder_layers[i, tail_span[i][0]:tail_span[i][1], :], dim = 0)[0] for i in range(batch_size)], dim = 0) # (batch_size, bert_embedding)
        pretrained_feat = torch.cat([head_entity_rep, tail_entity_rep], dim = 1) # (batch_size, 2 * bert_embedding)
        return pretrained_feat

    def forward(self, batch: Dict[str, torch.Tensor], mask: Optional[torch.BoolTensor]=None, msg: str='similarity', cut_gradient:bool=False):
        input_ids = batch['token_ids']
        input_mask = batch['attn_mask']
        head_span = batch['head_spans']
        tail_span = batch['tail_spans']

        if mask!= None:
            input_ids = input_ids[mask]
            input_mask = input_mask[mask]
            head_span = head_span[mask]
            tail_span = tail_span[mask]

        if msg == 'similarity':# used for centroid update 
            with torch.no_grad():
                pretrained_feat = self.get_pretrained_feature(input_ids, input_mask, head_span, tail_span) # (batch_size, 2 * bert_embedding)
            commonspace_rep = self.similarity_encoder(pretrained_feat) # (batch_size, keamns_dim)
            return commonspace_rep # (batch_size, keamns_dim)

        elif msg == 'reconstruct':
            with torch.no_grad():
                pretrained_feat = self.get_pretrained_feature(input_ids, input_mask, head_span, tail_span) # (batch_size, 2 * bert_embedding)
            commonspace_rep = self.similarity_encoder(pretrained_feat) # (batch_size, kmeans_dim)
            rec_rep = self.similarity_decoder(commonspace_rep) # (batch_size, 2 * bert_embedding)
            rec_loss = (rec_rep - pretrained_feat).pow(2).mean(-1)
            return commonspace_rep, rec_loss


        elif msg == 'labeled':
            pretrained_feat = self.get_pretrained_feature(input_ids, input_mask, head_span, tail_span) # (batch_size, 2 * bert_embedding)
            if cut_gradient:
                pretrained_feat = pretrained_feat.detach()
            logits = self.labeled_head(pretrained_feat) 
            return logits # (batch_size, num_class)

        elif msg == 'unlabeled':
            pretrained_feat = self.get_pretrained_feature(input_ids, input_mask, head_span, tail_span) # (batch_size, 2 * bert_embedding)
            if cut_gradient:
                pretrained_feat = pretrained_feat.detach()
            logits = self.unlabeled_head(pretrained_feat)
            return logits # (batch_size, new_class)

        else:
            raise NotImplementedError('not implemented!')
