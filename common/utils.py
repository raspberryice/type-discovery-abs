import os 
import json 
import re 
from typing import List
from copy import deepcopy 


import torch 
import numpy as np
from scipy.optimize import linear_sum_assignment

def clean_text(text: List[str]) -> List[str]: 
    ret = []
    for word in text:
        normalized_word = re.sub(u"([^\u0020-\u007f])", "", word)
        if normalized_word == '' or normalized_word == ' ' or normalized_word == '    ':
            normalized_word = '[UNK]'
        ret.append(normalized_word)
    return ret
   


def onedim_gather(src: torch.Tensor, dim: int, index: torch.LongTensor) -> torch.Tensor:
    '''
    src: (batch, M, L)
    index: (batch, M) or (batch, L) or (batch, 1)

    A version of the torch.gather function where the index is only along 1 dim.
    '''
    for i in range(len(src.shape)):
        if i!=0 and i!=dim:
            # index is missing this dimension
            index = index.unsqueeze(dim=i)
    target_index_size = deepcopy(list(src.shape))
    target_index_size[dim] = 1 
    index = index.expand(target_index_size)
    output = torch.gather(src, dim, index)
    return output 


def get_label_mapping(predicted_logits, labels):
    """
    Compute an assignment from predicted to labels.
    :param predicted_logits (N, n_classes)
    :param labels (N)
    """
    M = predicted_logits.size(1)
    predicted_numpy = torch.max(predicted_logits, dim=1)[1].detach().cpu().numpy()
    labels_numpy = labels.detach().cpu().numpy() 

    w = np.zeros((M, M)) # cost matrix
    for i in range(labels_numpy.size):
        w[ predicted_numpy[i], labels_numpy[i]] += 1

    mapping = linear_sum_assignment(w)
    map_matrix = torch.zeros((M, M), dtype=torch.float, device=labels.device)
    for i, j in np.transpose(np.asarray(mapping)):
        map_matrix[i,j] = 1 
    return mapping, map_matrix 



# From UNO/utils/eval.py
def cluster_acc(y_true:np.array, y_pred: np.array, reassign: bool=False):
    """
    Calculate clustering accuracy with assigment 

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64) # N*K
    y_pred = y_pred.astype(np.int64) # N*K
    assert y_pred.size == y_true.size # same number of clusters 

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64) # cost matrix 
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    

    if reassign: 
        mapping = compute_best_mapping(w)

        return sum([w[i, j] for i, j in mapping]) * 1.0 / y_pred.size
    else:
        acc= sum([w[i,i] for i in range(D)]) * 1.0/y_pred.size 
        return acc

def compute_best_mapping(w):
    return np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))


