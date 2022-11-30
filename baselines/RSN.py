'''
Implementation following https://github.com/thunlp/RSN/blob/master/RSN/model/siamodel.py 
'''
from typing import List, Dict, Tuple 


import torch 
import torch.nn as nn
import torch.nn.functional as F

from common.utils import onedim_gather 

# def _cnn_(cnn_input_shape,name=None):
    
#     convnet = Sequential()
#     convnet.add(Conv1D(230, 3,
#         input_shape = cnn_input_shape,
#         kernel_initializer = W_init,
#         bias_initializer = b_init_conv,
#         kernel_regularizer=l2(2e-4)
#         ))
#     convnet.add(MaxPooling1D(pool_size=cnn_input_shape[0]-4))
#     convnet.add(Activation('relu'))

#     convnet.add(Flatten())
#     convnet.add(Dense(cnn_input_shape[-1]*230, activation = 'sigmoid',
#         kernel_initializer = W_init,
#         bias_initializer = b_init_dense,
#         kernel_regularizer=l2(1e-3)
#         ))
#     return convnet

class ConvNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, 
            kernel_size=3)
        self.linear = nn.Linear(output_dim, output_dim)
        # TODO: kernel regularizer? 


    def forward(self, x: torch.FloatTensor)-> torch.FloatTensor:
        '''
        :params x: (batch, seq_len, input_dim)
        :return (batch, output_dim) 
        '''
        seq_len = x.size(1)
        x = x.transpose(1,2) 
        conv_output = self.conv(x) #(batch, output_dim, seq_len-2)
    
        pooled_output = torch.max_pool1d(conv_output, seq_len-2).squeeze(2) # (batch, output_dim)
        pooled_output = torch.relu(pooled_output)
        output = torch.sigmoid(self.linear(pooled_output))
        
        return output 



    
class RSNLayer(nn.Module):
    def __init__(self, input_dim, rel_dim, use_cnn:bool=True, dropout_p:float=0.1, max_len=300, pos_emb_dim=5) -> None:
        super().__init__()

        self.use_cnn = use_cnn 
        self.max_len = max_len 

        self.dropout = nn.Dropout(p=dropout_p)
        self.pos_emb = nn.Embedding(num_embeddings=max_len*2, embedding_dim= pos_emb_dim)

        if use_cnn:
            self.conv_net = ConvNet(input_dim +2*pos_emb_dim, rel_dim) 
        else:
            self.proj = nn.Linear(input_dim*2, rel_dim)
        self.p = nn.Linear(rel_dim, 1) 


    def get_pos_embedding(self, spans, indexes):
        '''
        :param spans: (B, 2)
        :param indexes: (B, seq_len)
        
        :return pos_embed: (B, seq_len, pos_emb_dim)
        '''
        pos1 = indexes - spans[:, 0].unsqueeze(1)
        pos_embed = self.pos_emb(pos1+self.max_len)
        return pos_embed 

    def embed(self, head_spans, tail_spans, seq_output):
        '''
        Used for prediction.

        :return x: (B, rel_dim)
        '''
        seq_len = seq_output.size(1)
        batch_size = seq_output.size(0)
        seq_output = self.dropout(seq_output)
        
        if self.use_cnn:
            # position embeddings 
            indexes = torch.arange(0, seq_len, dtype=torch.long, device=head_spans.device).repeat((batch_size, 1)) # B, seq_len 
            head_pos_embed = self.get_pos_embedding(head_spans, indexes)
            tail_pos_embed = self.get_pos_embedding(tail_spans, indexes)
            seq_output_with_pos = torch.concat([seq_output, head_pos_embed, tail_pos_embed], dim=2) 

            x = self.conv_net(seq_output_with_pos) #B, rel_dim
        else:
            head_rep = onedim_gather(seq_output, dim=1, index=head_spans[:, 0].unsqueeze(1))
            tail_rep = onedim_gather(seq_output, dim=1, index=tail_spans[:, 0].unsqueeze(1))
            feat = torch.cat([head_rep, tail_rep], dim = 2).squeeze(1) 
            x = self.proj(feat) # B, rel_dim 

        return x 

    def compute_distance(self, x1, x2):
        dis = torch.abs(x1 - x2) # (rel_dim)
        prob = torch.sigmoid(self.p(dis)) 
        return prob 


    def forward(self, head_spans: torch.LongTensor, tail_spans:torch.LongTensor, 
        seq_output: torch.FloatTensor, perturb:torch.FloatTensor=None) ->torch.FloatTensor:
        '''
        :param seq_output: (B, seq_len, input_dim)
        :return: (B, B)
        '''
        seq_len = seq_output.size(1)
        batch_size = seq_output.size(0)
        seq_output = self.dropout(seq_output)
        if perturb!=None:
            seq_output += perturb 
        
        if self.use_cnn:
            # position embeddings 
            indexes = torch.arange(0, seq_len, device=head_spans.device).repeat((batch_size, 1)) # B, seq_len 
            head_pos_embed = self.get_pos_embedding(head_spans, indexes)
            tail_pos_embed = self.get_pos_embedding(tail_spans, indexes)
            seq_output_with_pos = torch.concat([seq_output, head_pos_embed, tail_pos_embed], dim=2) 

            x = self.conv_net(seq_output_with_pos) #B, rel_dim
        else:
            head_rep = onedim_gather(seq_output, dim=1, index=head_spans[:, 0].unsqueeze(1))
            tail_rep = onedim_gather(seq_output, dim=1, index=tail_spans[:, 0].unsqueeze(1))
            feat = torch.cat([head_rep, tail_rep], dim = 2).squeeze(1) 
            x = self.proj(feat) # B, rel_dim 

        dis = torch.abs(x.unsqueeze(1) - x.unsqueeze(0)) # (B, B, rel_dim)
        pair_prob = torch.sigmoid(self.p(dis).squeeze(-1)) 

        return pair_prob 
