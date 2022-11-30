from typing import List, Optional, Tuple, Dict
from collections import OrderedDict 
from math import ceil
import torch
from torch import nn   
import torch.nn.functional as F 

class Prototypes(nn.Module):
    def __init__(self, feat_dim, num_prototypes, norm:bool=False):
        super().__init__()

        if norm:
            self.norm = nn.LayerNorm(feat_dim)
        else:
            self.norm = lambda x: x 
        self.prototypes = nn.Linear(feat_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def initialize_prototypes(self, centers):
        self.prototypes.weight.copy_(centers)
        return 
    

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def freeze_prototypes(self):
        self.prototypes.requires_grad_(False)
    
    def unfreeze_prototypes(self):
        self.prototypes.requires_grad_(True)

    
    def forward(self, x):
        x = self.norm(x)
        return self.prototypes(x)


class MLP(nn.Module):
    '''
    Simple n layer MLP with ReLU activation and batch norm.
    The order is Linear, Norm, ReLU 
    '''
    def __init__(self, feat_dim: int, hidden_dim: int,  latent_dim:int, 
        norm:bool=False, norm_type:str='batch', layers_n: int = 1, dropout_p: float =0.1 ):
        '''
        :param norm_type: one of layer, batch 
        '''
        super().__init__() 
        self.feat_dim= feat_dim 
        self._hidden_dim= hidden_dim
        self.latent_dim = latent_dim 
        self.input2hidden = nn.Linear(feat_dim, hidden_dim) 
        self.dropout = nn.Dropout(p=dropout_p)       
        layers = [self.dropout, ]
        for i in range(layers_n):
            if i==0:
                layers.append(nn.Linear(feat_dim, hidden_dim))
                out_dim = hidden_dim 
            elif i==1:
                layers.append(nn.Linear(hidden_dim, latent_dim))
                out_dim = latent_dim 
            else:
                layers.append(nn.Linear(latent_dim, latent_dim))
                out_dim = latent_dim 
            if norm:
                if norm_type == 'batch':
                    layers.append(nn.BatchNorm1d(out_dim))
                else:
                    layers.append(nn.LayerNorm(out_dim))
            if i < layers_n -1: # last layer has no relu 
                layers.append(nn.ReLU())
        
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        '''
        :param input: torch.FloatTensor (batch, ..., feat_dim) 

        :return output: torch.FloatTensor (batch, ..., hidden_dim)
        '''
        output = self.net(input.reshape(-1, self.feat_dim))
       
        original_shape = input.shape 
        new_shape = tuple(list(input.shape[:-1]) + [self.latent_dim])

        output = output.reshape(new_shape) 
        return output 

class ReconstructionNet(nn.Module):
    '''
    projection from hidden_size back to feature_size.
    '''
    def __init__(self, feature_size:int, hidden_size: int, latent_size:int ) -> None:
        super().__init__()
        self.feature_size = feature_size 
        self.hidden_size = hidden_size 
        assert (feature_size > hidden_size)
        self.latent_size = latent_size 
        self.net = nn.Sequential(
            nn.Linear(in_features=self.latent_size, out_features=self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.feature_size)
        )

    def forward(self, inputs: torch.FloatTensor):
        
        output = self.net(inputs.reshape(-1, self.hidden_size))
        new_shape = tuple(list(inputs.shape[:-1]) + [self.feature_size])

        output = output.reshape(new_shape) 
        return output 
    
class CommonSpaceCache(nn.Module):
    '''
    A cache for saving common space embeddings and using it to compute contrastive loss.
    '''
    def __init__(self, feature_size:int, known_cache_size: int, unknown_cache_size: int, metric_type: str='cosine', sim_thres: float=0.8) -> None:
        super().__init__()
        self.feature_size = feature_size 
        self.known_cache_size = known_cache_size
        self.unknown_cache_size = unknown_cache_size
        self.known_len=0
        self.unknown_len =0 

        self.metric_type =metric_type 
        self.metric = nn.CosineSimilarity(dim=2, eps=1e-8)
        self.sim_thres=sim_thres 
       
        self.temp = 0.1 # temperature for softmax 

        self.register_buffer("known_cache", torch.zeros((known_cache_size, feature_size), dtype=torch.float), persistent=False)
        self.register_buffer("unknown_cache", torch.zeros((unknown_cache_size, feature_size), dtype=torch.float), persistent=False)

        self.register_buffer("known_labels", torch.zeros((known_cache_size,), dtype=torch.long), persistent=False)
        self.register_buffer("unknown_labels", torch.zeros((unknown_cache_size, ), dtype=torch.long), persistent=False)


    def cache_full(self)-> bool:
        if (self.known_len == self.known_cache_size) and (self.unknown_len == self.unknown_cache_size):
            return True 
        else:
            return False 

    @torch.no_grad() 
    def update_batch(self, embeddings: torch.FloatTensor, known_mask: torch.BoolTensor, labels: Optional[torch.LongTensor]=None) -> None:
        '''
        Add embeddings to cache.
        :param embeddings: (batch, feature_size)
        '''
        embeddings_detached = embeddings.detach() 

        known_embeddings = embeddings_detached[known_mask,:]
        known_size = known_embeddings.size(0)
        new_known_cache = torch.concat([known_embeddings, self.known_cache], dim=0)
        self.known_cache = new_known_cache[:self.known_cache_size]
        self.known_len = min(self.known_len + known_size, self.known_cache_size)
        if labels!=None: 
            known_labels = labels[known_mask] 
            self.known_labels = torch.concat([known_labels, self.known_labels], dim=0)[:self.known_cache_size]
            unknown_labels = labels[~known_mask]
            self.unknown_labels = torch.concat([unknown_labels, self.unknown_labels], dim=0)[:self.unknown_cache_size]


        unknown_embeddings = embeddings_detached[~known_mask,: ]
        unknown_size = unknown_embeddings.size(0)
        new_unknown_cache = torch.concat([unknown_embeddings, self.unknown_cache], dim=0)
        self.unknown_cache = new_unknown_cache[:self.unknown_cache_size]
        self.unknown_len = min(self.unknown_len + unknown_size, self.unknown_cache_size)
        return 

    @torch.no_grad()
    def get_positive_example(self, embedding: torch.FloatTensor, known: bool =False) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        '''
        :param embeddings (N, feature_dim)

        :returns (N, feature_dim)
        '''
        embedding_detached = embedding.detach() 
        if known: 
            cache = self.known_cache
            label_cache = self.known_labels
        else:
            cache = self.unknown_cache 
            label_cache = self.unknown_labels

        if self.metric_type == 'cosine':       
            similarity = self.metric(embedding_detached.unsqueeze(dim=1), cache.unsqueeze(dim=0)) # N, cache_size
        else:
            similarity = torch.einsum("ik,jk->ij", embedding_detached, cache) 
        
        max_sim, max_idx = torch.max(similarity, dim=1) #(N, )
        min_thres = self.sim_thres 
        valid_pos_mask = (max_sim > min_thres) #(N, )
        pos_embeddings = cache[max_idx, :] # (N, feature_dim)
        pos_labels = label_cache[max_idx] # (N, )

        return pos_embeddings, valid_pos_mask, pos_labels 
    
    @torch.no_grad() 
    def get_negative_example_for_unknown(self, embedding: torch.FloatTensor, k: int=3) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
        Take half of the negative examples from the unknown cache and half from the known cache.
        :param embeddings (N, feature_dim)
        '''
        embedding_detached= embedding.detach() 
        N = embedding_detached.size(0)
        if self.metric_type == 'cosine':
            unknown_similarity = self.metric(embedding_detached.unsqueeze(dim=1), self.unknown_cache.unsqueeze(dim=0)) # N, cache_size
        else:
            unknown_similarity = torch.einsum('ik,jk->ij', embedding_detached, self.unknown_cache)
        
        sorted_unk_idx = torch.argsort(unknown_similarity, dim=1) # N, cache_size
        unk_n = ceil(sorted_unk_idx.size(1) /2)
        candidate_neg_unk_idx = sorted_unk_idx[:, :unk_n] # N, cache_size/2
        # this is used for generating indexes 
        neg_unk_list = []
        for i in range(N):
            random_idx = torch.randperm(n=unk_n, dtype=torch.long, device=embedding.device)[:k]
            chosen_neg_unk_idx = candidate_neg_unk_idx[i, :][random_idx]
            chosen_neg_unk = self.unknown_cache[chosen_neg_unk_idx, :] # K, feature_size 
            neg_unk_list.append(chosen_neg_unk)
        
        if self.metric_type == 'cosine':
            known_similarity = self.metric(embedding_detached.unsqueeze(dim=1), self.known_cache.unsqueeze(dim=0)) # (N, cache_size)
        else:
            known_similarity = torch.einsum("ik,jk->ij", embedding_detached, self.known_cache)
        
        sorted_known_idx = torch.argsort(known_similarity, dim=1, descending=True) # choose hard examples (N, cache_size)
        neg_known_list = []
        chosen_neg_known_idx = sorted_known_idx[:, :k]
        for i in range(N):
            chosen_neg_known = self.known_cache[chosen_neg_known_idx[i], :]
            neg_known_list.append(chosen_neg_known)

        neg_unk = torch.stack(neg_unk_list, dim=0)
        neg_known = torch.stack(neg_known_list, dim=0) # (N, K, feature_size)
            
        return neg_unk, neg_known

    def get_contrastive_candidates(self, embeddings: torch.FloatTensor, neg_n: int=6, labels: Optional[torch.LongTensor]=None):
        N = embeddings.size(0)
        if labels!=None: assert (labels.size(0) == N)

        pos_embeddings, valid_pos_mask, pos_labels  = self.get_positive_example(embeddings, known=False) # (N, hidden_dim)
        assert (pos_embeddings.shape == embeddings.shape )
        # report positive sample accuracy 
        pos_acc = self.compute_accuracy(labels[valid_pos_mask], pos_labels[valid_pos_mask])

        neg_unk_embeddings, neg_known_embeddings  = self.get_negative_example_for_unknown(embeddings, k=ceil(neg_n/2)) # (N, K, hidden_dim)
        candidates = torch.concat([pos_embeddings.unsqueeze(dim=1), neg_unk_embeddings, neg_known_embeddings], dim=1) # (N, 2K+1, hidden_dim)
        # scores = torch.einsum('ik,ijk->ij', embeddings, candidates) # (N, 2K+1 )
        # targets = torch.zeros((N,), dtype=torch.long, device=scores.device)
        # loss = F.cross_entropy(scores/self.temp, targets)
        return candidates, valid_pos_mask, pos_acc 

    def compute_accuracy(self, labels, other_labels):
        # consider moving average 
        assert (labels.shape == other_labels.shape)
        acc = torch.sum(labels == other_labels)*1.0 / labels.size(0)
        return acc 


class ClassifierHead(nn.Module):
    def __init__(self, args, feature_size: int, 
            n_classes: int, layers_n: int = 1, 
            n_heads: int =1, dropout_p: float =0.2, hidden_size: Optional[int]=None) -> None:
        super().__init__()
        self.args = args 

        self.feature_size = feature_size 
        self.n_classes = n_classes 
        self.n_heads = n_heads
        if hidden_size: 
            self.hidden_size = hidden_size 
        else:
            self.hidden_size = feature_size 

        if layers_n == 1:
            self.classifier = nn.Sequential(OrderedDict(
                [('dropout',nn.Dropout(p=dropout_p)),
                ('centroids', Prototypes(feat_dim=self.hidden_size, num_prototypes=self.n_classes))]
                ))
        elif layers_n > 1:
            self.classifier = nn.Sequential(OrderedDict(
                [('mlp', MLP(feat_dim=self.feature_size, hidden_dim=self.hidden_size, latent_dim=self.hidden_size, norm=True,layers_n=layers_n-1)),
                ('dropout',nn.Dropout(p=dropout_p)),
                ('centroids', Prototypes(feat_dim=self.hidden_size, num_prototypes=self.n_classes))]
                ))

    def initialize_centroid(self, centers):
        for n, module in self.classifier.named_modules():
            if n=='centroids':
                module.initialize_prototypes(centers)
        
        return 

    def update_centroid(self):
        '''
        The centroids are essentially just the vectors in the final Linear layer. Here we normalize them. they are trained along with the model.
        '''
        for n, module in self.classifier.named_modules():
            if n=='centroids':
                module.normalize_prototypes()

        return 
    
    def freeze_centroid(self):
        '''
        From Swav paper, freeze the prototypes to help with initial optimization.
        '''
        for n, module in self.classifier.named_modules():
            if n=='centroids':
                module.freeze_prototypes() 

        return 

    def unfreeze_centroid(self):
        for n, module in self.classifier.named_modules():
            if n=='centroids':
                module.unfreeze_prototypes() 

        return 


    def forward(self, inputs: torch.FloatTensor):
        '''
        :params inputs: (batch, feat_dim)

        :returns logits: (batch, n_classes)
        '''
        outputs = self.classifier(inputs)
        return outputs 
