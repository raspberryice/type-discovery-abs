from collections import defaultdict, Counter
from typing import Dict, Tuple, List,Optional
import json 
import os 

import torch 



class PairwiseClusterPredictionsWrapper(torch.nn.Module):
    def __init__(self, prefix: str='', task: str='rel') -> None:
        super().__init__()
        self.prefix = prefix 
        self.task = task 

        self.predictions = {} 

    def on_epoch_end(self, pred_cluster:Dict, metadata:Dict)->None:

        for uid in pred_cluster:
            cluster_idx = pred_cluster[uid]
            target_idx = metadata[uid]['label_idx']
            sentence = ' '.join(metadata[uid]['tokens'])
            label = metadata[uid]['label'] # type: str
            self.predictions[uid] = {
                'uid': uid, 
                'sentence': sentence,
                'label': label,
                'cluster_idx': cluster_idx,
                'target_idx': target_idx
            }
            if self.task == 'rel':
                self.predictions[uid]['subj'] = metadata[uid]['subj']
                self.predictions[uid]['obj'] = metadata[uid]['obj']
            elif self.task == 'event':
                self.predictions[uid]['trigger'] = metadata[uid]['trigger']
        

        # index by cluster 
        self.clusters = defaultdict(list)
        for uid, d in self.predictions.items():
            cluster_idx = d['cluster_idx']
            self.clusters[cluster_idx].append(d)
        return 

    
    def save(self, ckpt_dir: str):
        with open(os.path.join(ckpt_dir, f'{self.prefix}_predictions.json'),'w') as f:
            json.dump(self.predictions, f, indent=2)

        with open(os.path.join(ckpt_dir, f'{self.prefix}_clusters.json'),'w') as f:
            json.dump(self.clusters, f, indent=2)



class ClusterPredictionsWrapper(torch.nn.Module):
    def __init__(self, reassign: bool=False, prefix: str='', known_classes: int=31, 
        task: str='rel', save_names: bool =False) -> None:
        super().__init__()
        self.reassign = reassign
        self.prefix = prefix 
        self.known_classes=known_classes
        self.task = task 
        self.save_names = save_names 

        self.pred_cluster_cache = []
        self.target_cluster_cache = []
        self.pred_prob_cache = []

        self.predictions = {} # uid -> {tokens, label, cluster_idx}
    
    def update_batch(self, meta: List[Dict], logits: torch.FloatTensor, 
        targets: torch.LongTensor, incremental:bool=False, names: Optional[List[str]]=None)->None:
        '''
        :param logits: (batch, n_cluster)
        :param targets: (batch)
        '''
        
        if not incremental: 
            assert (targets.min() >= self.known_classes)
            # consider only unknown classes 
            prob = torch.softmax(logits[:, self.known_classes:], dim=1)
            pred_prob, pred_cluster = torch.max(prob, dim=1) 
            self.pred_cluster_cache.append(pred_cluster)
            self.target_cluster_cache.append(targets-self.known_classes)
            self.pred_prob_cache.append(pred_prob) 
        else:
            prob = torch.softmax(logits, dim=1) 
            pred_prob, pred_cluster = torch.max(prob, dim=1)
            self.target_cluster_cache.append(targets)
            self.pred_prob_cache.append(pred_prob)
        
        batch_size = logits.size(0)
        for i in range(batch_size):
            uid = meta[i]['uid']
            sentence = ' '.join(meta[i]['tokens'])
            cluster_idx = pred_cluster[i].item() 
            target_idx = targets[i].item()
            self.predictions[uid] = {
                'uid': uid, 
                'sentence': sentence,
                'label': meta[i]['label'],
                'cluster_idx': cluster_idx,
                'target_idx': target_idx,
                'prob': pred_prob[i].item()
            }

            if names!= None:
                self.predictions[uid]['names'] = names[i].split() 
            else:
                self.predictions[uid]['names'] = [] 
            if self.task == 'rel':
                self.predictions[uid]['subj'] = meta[i]['subj']
                self.predictions[uid]['obj'] = meta[i]['obj']
            elif self.task == 'event':
                self.predictions[uid]['trigger'] = meta[i]['trigger']


        return 

    def on_epoch_end(self):
        # index by cluster 
        self.clusters = defaultdict(list)
        cluster_scores = defaultdict(list) # cluster_id -> List[(uid, prob)]
        cluster_names = defaultdict(Counter) # cluster_id -> counter  
        for uid, d in self.predictions.items():
            cluster_idx = d['cluster_idx']
            cluster_scores[cluster_idx].append((uid, d['prob'])) 
            cluster_names[cluster_idx].update(d['names'])
        
        self.cluster_freq_names = {} 
        # sort by score 
        for cluster_idx in cluster_scores.keys():
            uid_sorted = sorted(cluster_scores[cluster_idx], key=lambda x: x[1], reverse=True)
            for tup in uid_sorted:
                uid = tup[0]
                d = self.predictions[uid]
                self.clusters[cluster_idx].append(d)
                self.cluster_freq_names[cluster_idx] = cluster_names[cluster_idx].most_common(n=10) # List (str, int)

    def save(self, ckpt_dir: str):
        with open(os.path.join(ckpt_dir, f'{self.prefix}_predictions.json'),'w') as f:
            json.dump(self.predictions, f, indent=2)

        with open(os.path.join(ckpt_dir, f'{self.prefix}_clusters.json'),'w') as f:
            json.dump(self.clusters, f, indent=2)


        if self.save_names:
            with open(os.path.join(ckpt_dir, f'{self.prefix}_cluster_names.json'),'w') as f:
                json.dump(self.cluster_freq_names, f, indent=2) 