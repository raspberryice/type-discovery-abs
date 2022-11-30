from collections import defaultdict
from typing import Dict, Tuple, List
import json 
import os 

import torch 
import numpy as np

from sklearn.metrics.cluster import homogeneity_completeness_v_measure, adjusted_rand_score, normalized_mutual_info_score,fowlkes_mallows_score
from sklearn.cluster import SpectralClustering
from torchmetrics import Metric, MetricCollection
import networkx as nx
import community

from common.utils import cluster_acc
from common.b3 import calc_b3
import common.log as log
logger = log.get_logger('root')
'''
Clustering metrics: 
B-cube 
V-measure
Adjusted Rand Index 
'''


class PairwiseClusteringMetricsWrapper(torch.nn.Module):
    def __init__(self, stage: str='train',  prefix: str='',
         cluster_n: int=10, clustering_method: str='louvain') -> None:
        '''
        :param clustering method: str, one of louvain, hac
        '''
        super().__init__()
        self.stage = stage 
        self.prefix = prefix 
       
        self.target_cluster_cache = []
        self.metrics = None 
        self.clustering_method = clustering_method 
        self.cluster_n = cluster_n 

    def get_pred_cluster(self, pred_metric:Dict, labels: Dict, 
        clustering_method: str, merge_iso:bool=True, iso_thres:int=5 )->Dict:
        '''
        :param pred_metric: This is actually a distance, the smaller the metric, the closer
        '''
        N = len(labels)
        uid2idx = {k:idx for idx, k in enumerate(labels.keys())}
        idx2uid = {v:k for k,v in uid2idx.items()}
        if clustering_method == 'louvain':
            g = nx.Graph() 
            g.add_nodes_from(labels.keys()) 

            for k, v in pred_metric.items():
                if round(v) == 0:
                    g.add_edge(k[0], k[1])
        
            partition = community.best_partition(g) 
            pred_cluster = {k: partition[k] for k in labels}  
        elif clustering_method == 'spectral':
            sim_matrix = np.zeros((N, N))
            for idx in range(N):
                for other_idx in range(idx, N):
                    key = (idx2uid[idx], idx2uid[other_idx])
                    rev_key = (idx2uid[other_idx], idx2uid[idx])
                    sim_matrix[idx, other_idx] = 1- pred_metric[key] # dist to similarity 
                    sim_matrix[other_idx, idx] = 1- pred_metric[rev_key] 

            clustering = SpectralClustering(n_clusters=self.cluster_n, affinity='precomputed', assign_labels='discretize').fit(sim_matrix)
            pred_cluster = {idx2uid[idx]: int(clustering.labels_[idx]) for idx in range(N)}

        else:
            raise NotImplementedError(f'clustering method {clustering_method} not supported.')
        if merge_iso:
            cluster2uid = defaultdict(list)
            for k,v in pred_cluster.items():
                cluster2uid[v].append(k)
            cluster_sizes = {k: len(v) for k,v in cluster2uid.items()}
            iso_clus = [k for k in cluster_sizes if cluster_sizes[k] <=iso_thres] 

            # reassign the data points in the small clusters 
            for clus_idx in iso_clus:
                for uid in cluster2uid[clus_idx]:
                    all_dist = np.zeros((N,), dtype=np.float32) 
                    for other_uid in labels.keys():
                        all_dist[uid2idx[other_uid]] =  pred_metric[(uid, other_uid)]
                    search_idx_list = np.argsort(all_dist)
                    for other_idx in search_idx_list:
                        other_uid = idx2uid[other_idx]
                        if pred_cluster[other_uid] not in iso_clus:
                            pred_cluster[uid] = pred_cluster[other_uid]
                            break   


        return pred_cluster  
    
    def on_epoch_end(self, pred_metric, labels)->Tuple[Dict[str, float], Dict[str, int]]:
        pred_cluster = self.get_pred_cluster(pred_metric, labels, self.clustering_method)
        # convert dict to numpy array 
        pred_cluster_list = []
        target_cluster_list = []
        for k,v in pred_cluster.items():
            pred_cluster_list.append(v)
            target_cluster_list.append(labels[k]) 

        all_pred_cluster = np.array(pred_cluster_list) 
        all_target_cluster = np.array(target_cluster_list) 
        
        
        b3_metrics = calc_b3(np.expand_dims(all_pred_cluster,axis=0), np.expand_dims(all_target_cluster, axis=0))
        ari = adjusted_rand_score(all_target_cluster, all_pred_cluster)
        v_hom, v_comp, v_f1 = homogeneity_completeness_v_measure(all_target_cluster, all_pred_cluster)
        nmi = normalized_mutual_info_score(all_target_cluster, all_pred_cluster)
        fm = fowlkes_mallows_score(all_target_cluster, all_pred_cluster)
        metrics = {
            'b3_f1': b3_metrics[0],
            'b3_prec': b3_metrics[1],
            'b3_recall': b3_metrics[2],
            'ARI': ari,
            'homogeneity': v_hom,
            'completeness': v_comp,
            'v_measure': v_f1,
            'NMI': nmi,
            'fowlkes_mallows': fm 
        }

        acc = cluster_acc(all_target_cluster, all_pred_cluster, reassign=True)

        metrics['acc'] = float(acc)  
        self.metrics = metrics 

        return metrics, pred_cluster 


    
    def save(self, ckpt_dir: str):
        if self.metrics: 
            metrics = self.metrics 
        else:    
            metrics = self.on_epoch_end()

        output_dict = {k: v for k,v in metrics.items()}
        with open(os.path.join(ckpt_dir, f'{self.prefix}_metrics.json'),'w') as f:
            json.dump(output_dict, f, indent=2)

        return 

class ClusteringMetricsWrapper(torch.nn.Module):
    def __init__(self, stage: str='train', known: bool=False,  prefix: str='', known_classes:int=31) -> None:
        '''
        :param reassign: whether to compute the cluster assignment. Set to true for unknown classes.
        '''
        super().__init__()
        self.stage = stage 
        self.prefix = prefix 
        self.known_classes = known_classes
        self.known = known 
        self.reassign = False if known else True 

        self.pred_cluster_cache = []
        self.target_cluster_cache = []

        self.metrics = None 

    def reset(self):
        self.pred_cluster_cache = []
        self.target_cluster_cache = []

    def update_batch(self, logits: torch.FloatTensor, targets: torch.LongTensor, incremental:bool=False )->None:
        '''
        :param logits: (batch, n_cluster)
        :param targets: (batch)
        '''
        if not incremental:
            if not self.known: 
                # consider only unknown classes 
                _, pred_cluster = torch.max(logits[:, self.known_classes:], dim=1) 
                self.pred_cluster_cache.append(pred_cluster)
                self.target_cluster_cache.append(targets-self.known_classes)
            else:
                # consider only known classes 
                _, pred_cluster = torch.max(logits[:, :self.known_classes], dim=1) 
                self.pred_cluster_cache.append(pred_cluster)
                self.target_cluster_cache.append(targets)
        else:
            _, pred_cluster = torch.max(logits, dim=1)
            self.pred_cluster_cache.append(pred_cluster)
            self.target_cluster_cache.append(targets)
        return 


    def on_epoch_end(self)-> Dict[str, float]:
        all_pred_cluster = torch.concat(self.pred_cluster_cache).cpu().numpy() 
        all_target_cluster = torch.concat(self.target_cluster_cache).cpu().numpy()
        b3_metrics = calc_b3(np.expand_dims(all_pred_cluster,axis=0), np.expand_dims(all_target_cluster, axis=0))
        ari = adjusted_rand_score(all_target_cluster, all_pred_cluster)
        v_hom, v_comp, v_f1 = homogeneity_completeness_v_measure(all_target_cluster, all_pred_cluster)
        nmi = normalized_mutual_info_score(all_target_cluster, all_pred_cluster)
        fm = fowlkes_mallows_score(all_target_cluster, all_pred_cluster)
        metrics = {
            'b3_f1': b3_metrics[0],
            'b3_prec': b3_metrics[1],
            'b3_recall': b3_metrics[2],
            'ARI': ari,
            'homogeneity': v_hom,
            'completeness': v_comp,
            'v_measure': v_f1,
            'NMI': nmi,
            'fowlkes_mallows': fm 
        }

        acc = cluster_acc(all_target_cluster, all_pred_cluster, reassign=self.reassign)

        metrics['acc'] = float(acc)  
        self.metrics = metrics 

        return metrics 


    
    def save(self, ckpt_dir: str):
        if self.metrics: 
            metrics = self.metrics 
        else:    
            metrics = self.on_epoch_end()

        output_dict = {k: v for k,v in metrics.items()}
        with open(os.path.join(ckpt_dir, f'{self.prefix}_metrics.json'),'w') as f:
            json.dump(output_dict, f, indent=2)

        return 

class PsuedoLabelMetricWrapper(torch.nn.Module):
    def __init__(self, prefix: str='', cache_size:int = 1024, known_classes: int = 31, unknown_classes: int =10) -> None:
        super().__init__()

        self.prefix = prefix 
        self.cache_size = cache_size 
        self.known_classes= known_classes
        self.unknown_classes = unknown_classes

        self.register_buffer('predicted_cache', torch.zeros((cache_size), dtype=torch.long), persistent=False)
        self.register_buffer('label_cache', torch.zeros((cache_size), dtype=torch.long), persistent=False)
        self.cur_len =0 


    def compute_metric(self):
        # this operation will slow down the training, use only for diagnosis.
        label_array = self.label_cache[:self.cur_len].cpu().numpy() 
        predicted_array = self.predicted_cache[:self.cur_len].cpu().numpy() 
        acc = cluster_acc(label_array,predicted_array, reassign=True)
        ari = adjusted_rand_score(label_array, predicted_array)
        return acc, ari


    def update_batch(self, psuedo_labels: torch.FloatTensor, labels: torch.LongTensor):
        '''
        :param pl: (batch, M+N)
        :param labels: (batch) over M+N
        '''
        batch_size = psuedo_labels.size(0)

        _, predicted = torch.max(psuedo_labels[:, self.known_classes:], dim=1)
        new_cache = torch.concat([predicted, self.predicted_cache], dim=0)
        self.predicted_cache = new_cache[:self.cache_size]
        self.cur_len += batch_size
        self.cur_len = min(self.cur_len, self.cache_size)

        new_label_cache = torch.concat([labels - self.known_classes, self.label_cache], dim=0)
        self.label_cache = new_label_cache[:self.cache_size]

        return

