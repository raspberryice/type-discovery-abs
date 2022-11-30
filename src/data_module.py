import json
import os 
from collections import defaultdict, Counter  
import random


from typing import List, Dict, Tuple, Callable, Union, Optional
import argparse
from math import ceil 
from copy import deepcopy
import pickle 
from itertools import cycle 
import numpy as np 

from torch.utils.data import DataLoader, ConcatDataset, Dataset, IterableDataset, WeightedRandomSampler
import torch  
import pytorch_lightning as pl 
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm 


from common.utils import clean_text, cluster_acc
import common.log as log 

logger = log.get_logger('root')

class InputExample(object):
  def __init__(self, unique_id: Union[int, str] , text: List[str], 
        head_span: List[int], tail_span: List[int], 
        label: int, relation_name: str, known: bool=True):

    self.uid = unique_id
    self.tokens = text # type: List[str]
    self.head_span = head_span
    self.tail_span = tail_span
    self.ori_label = label 
    self.label = label
    self.known = known 
    self.relation_name = relation_name 
    self.pseudo_label = -1

def reverse_ex(ex:InputExample, known_types:int, unknown_types:int, known:bool=True ) -> InputExample:
    rev_label = ex.ori_label + known_types
    rev_relation_name = 'rev_'+ ex.relation_name 

    return InputExample(ex.uid, ex.tokens, ex.tail_span, ex.head_span, rev_label, rev_relation_name, ex.known)

class RelationDict(object):
    '''
    Bidirectional dictionary for relations to int.
    '''
    def __init__(self) -> None:
        self.rel2int = {} 
        self.int2rel = {} 

        self.qnode2int = {} 


    def add(self, relation: str, id: Optional[id]=None, qnode: Optional[str]=None) -> None:
        if id: rel_id = id 
        else: rel_id = len(self.rel2int)
        self.rel2int[relation] = rel_id 
        self.int2rel[rel_id] = relation 
        self.qnode2int[qnode] = rel_id 
        return 


    def get_id(self, relation: str) -> int: 
        return self.rel2int[relation]

    def get_id_from_qnode(self, qnode:str) -> int:
        return self.qnode2int[qnode]


    def get_relation(self, id:int) -> str:
        return self.int2rel[id]

    def __len__(self): 
        return len(self.rel2int)


           

def _load_relation_dict(relation_file_path: str, has_qnode: bool=False)-> RelationDict:
    '''
    format: relation_name\tnum_of_instances 
    '''
    relation_dict = RelationDict()

    with open(relation_file_path) as f:
        for lidx, line in enumerate(f):
            fields = line.strip().split("    ")
            if len(fields) == 0: break # trailing new line 
            relation_name  = fields[0]
            # instance_n = fields[1] for tacred 
            # relation_desc = fields[1] for fewrel
            if has_qnode: 
                qnode = relation_name.split(':')[0]
                relation_dict.add(relation_name, lidx, qnode)
            else:
                relation_dict.add(relation_name, lidx)
                
    
    return relation_dict 

def _create_example_fewrel(instance:Dict, rel_id: int, rel_name: str, idx: int)-> InputExample:
    '''
    Convert a FewRel instance into InputExample class.
    ''' 
    text = clean_text(instance['tokens'])
    head_span = [instance['h'][2][0][0], instance['h'][2][0][-1]]
    tail_span = [instance['t'][2][0][0], instance['t'][2][0][-1]]
    uid = f'{rel_name}_{idx}' # type: str

    input_example = InputExample(uid, text, head_span, tail_span, rel_id, rel_name)
    return input_example 


def _create_example_tacred(sample:Dict, relation_dict:RelationDict, max_word_length: int=80)-> InputExample:
    '''
    Convert a TACRED instance into InputExample class.

    The 'subj_end' and 'obj_end' indexes are contained in the entity span.

    Part of this code is from RoCORE. 
    '''
    text = clean_text(sample['tokens']) # remove non-ascii

    head_span = [sample['subj_start'], sample['subj_end']]
    tail_span = [sample['obj_start'], sample['obj_end']]
    if len(text) >= max_word_length:
        if head_span[1] < tail_span[0]:
            text = text[head_span[0]:tail_span[1]+1]
            num_remove = head_span[0]

        else:
            text = text[tail_span[0]:head_span[1]+1]
            num_remove = tail_span[0]
        head_span = [head_span[0]-num_remove, head_span[1]-num_remove]
        tail_span = [tail_span[0]-num_remove, tail_span[1]-num_remove]
    
    relation_id = relation_dict.get_id(sample['relation'])
    relation_name = sample['relation']
    uid = sample['id'] # type: str

    input_example = InputExample(uid, text, head_span, tail_span, relation_id, relation_name)
    return input_example 


def _convert_example_to_tok_feature(ex: InputExample, tokenizer:PreTrainedTokenizer) -> Dict:
    tokens = ex.tokens 
    subj_tokens = tokens[ex.head_span[0]: ex.head_span[1]+1 ]
    obj_tokens = tokens[ex.tail_span[0]: ex.tail_span[1]+1 ]

    meta = {
        'uid': ex.uid,
        'tokens': ex.tokens,
        'known': ex.known,
        'label': ex.relation_name,
        'feature_type': 'token',
        'subj': ' '.join(subj_tokens),
        'obj': " ".join(obj_tokens)
    }

    # insert entity markers 
    input_tokens = deepcopy(tokens)
    if ex.head_span[0] < ex.tail_span[0]:
        input_tokens.insert(ex.head_span[0], '<h>')
        input_tokens.insert(ex.head_span[1]+2, '</h>')

        input_tokens.insert(ex.tail_span[0]+2, '<t>')
        input_tokens.insert(ex.tail_span[1]+4, '</t>')
        # get the head span and tail span in bpe offset 
        substart = tokenizer.encode(' '.join(input_tokens[:ex.head_span[0] +1])) 
        subend = tokenizer.encode(' '.join(input_tokens[:ex.head_span[1]+2]))
        head_span = (len(substart) -1, len(subend) -1)
        objstart = tokenizer.encode(' '.join(input_tokens[:ex.tail_span[0]+3]))
        objend = tokenizer.encode(' '.join(input_tokens[:ex.tail_span[1]+4])) 
        tail_span = (len(objstart) -1, len(objend) -1)
    else:
        input_tokens.insert(ex.tail_span[0], '<t>')
        input_tokens.insert(ex.tail_span[1]+2, '</t>')
        input_tokens.insert(ex.head_span[0]+2, '<h>')
        input_tokens.insert(ex.head_span[1] + 4, '</h>')
        start1 = tokenizer.encode(' '.join(input_tokens[:ex.tail_span[0] +1])) 
        end1 = tokenizer.encode(' '.join(input_tokens[:ex.tail_span[1]+2]))
        tail_span = (len(start1) -1 , len(end1)- 1) # account for the ending [sep] token
        start2 = tokenizer.encode(' '.join(input_tokens[:ex.head_span[0]+3]))
        end2 = tokenizer.encode(' '.join(input_tokens[:ex.head_span[1]+4])) 
        head_span = (len(start2) -1, len(end2) -1)

    sentence = ' '.join(input_tokens)

    token_ids = tokenizer.encode(sentence, return_tensors='pt').squeeze(0) # (seq_len) 
    seq_len = token_ids.size(0)
    attn_mask = torch.ones((seq_len))


    # assert (tokenizer.decode(token_ids[head_span[0]:head_span[1]]) == ' '.join(subj_tokens))
    return {
        'meta': meta,
        'token_ids': token_ids,
        'attn_mask': attn_mask,
        'head_span': head_span,
        'tail_span': tail_span, 
        'label': ex.label,
        'known': ex.known,
        'pseudo_label': ex.pseudo_label
    }

def _convert_example_to_mask_feature(ex: InputExample, tokenizer: PreTrainedTokenizer, prompt_idx: int =0) -> Dict: 
    tokens = ex.tokens
    subj_tokens = tokens[ex.head_span[0]: ex.head_span[1]+1 ]
    obj_tokens = tokens[ex.tail_span[0]: ex.tail_span[1]+1 ]
    meta = {
        'uid': ex.uid,
        'tokens': ex.tokens,
        'known': ex.known,
        'label': ex.relation_name,
        'feature_type': 'mask',
        'subj': ' '.join(subj_tokens),
        'obj': " ".join(obj_tokens)
    }

    input_tokens = deepcopy(tokens)
    if ex.head_span[0] < ex.tail_span[0]:
        input_tokens.insert(ex.head_span[0], '<h>')
        input_tokens.insert(ex.head_span[1]+2, '</h>')

        input_tokens.insert(ex.tail_span[0]+2, '<t>')
        input_tokens.insert(ex.tail_span[1]+4, '</t>')
    else:
        input_tokens.insert(ex.tail_span[0], '<t>')
        input_tokens.insert(ex.tail_span[1]+2, '</t>')
        input_tokens.insert(ex.head_span[0]+2, '<h>')
        input_tokens.insert(ex.head_span[1] + 4, '</h>')

    if prompt_idx == 0:
        # obj is the <mask> of subj
        prompt = obj_tokens + ['is','the', tokenizer.mask_token, 'of']  + subj_tokens 
        mask_word_idx = len(input_tokens) + len(obj_tokens) + 2 # 1 for sep token 
        mask_word_prefix = input_tokens + obj_tokens + ['is', 'the']
    elif prompt_idx == 1:
        # the relation between <subj> and <obj> is <mask>
        prompt = ['the', 'relation', 'between'] + subj_tokens + ['and'] + obj_tokens + ['is', tokenizer.mask_token]
        mask_word_idx = len(input_tokens) + len(prompt) -1 
        mask_word_prefix = input_tokens + prompt[:1]

    prefix_bpe = tokenizer.encode(' '.join(mask_word_prefix))
    mask_bpe_idx = len(prefix_bpe) -1 

    token_ids = tokenizer.encode( ' '.join(input_tokens + prompt), return_tensors='pt').squeeze(0)
    seq_len = token_ids.size(0)
    attn_mask = torch.ones((seq_len), dtype=torch.bool)

    
    return {
        'meta': meta, 
        'token_ids': token_ids,
        'attn_mask': attn_mask,
        'mask_bpe_idx': mask_bpe_idx,
        'label': ex.label,
        'known': ex.known,
        'pseudo_label': ex.pseudo_label
    }


def batch_var_length(tensors: List[torch.Tensor], max_length: int =300):
    batch_size = len(tensors)
    pad_len = min(max_length, max([t.size(0) for t in tensors])) 
    batch_tensors = torch.zeros((batch_size, pad_len)).type_as(tensors[0])
    for i in range(batch_size):
        actual_len = min(pad_len, tensors[i].size(0))
        batch_tensors[i, :actual_len] = tensors[i][:actual_len]
    
    return batch_tensors 


class IterableMixedBatchDataset(IterableDataset):
    def __init__(self, args, tokenizer: PreTrainedTokenizer, 
        known_exs: List[InputExample], unknown_exs: List[InputExample], 
        mix_ratio: float, feature: str='token', seed: int =0) -> None:
        super().__init__()
        self.args = args 
        self.tokenizer = tokenizer 
        self.mix_ratio = mix_ratio 

        if feature == 'token': 
            self.feature_func = lambda x, t: [_convert_example_to_tok_feature(x,t ),_convert_example_to_tok_feature(x,t )]
        elif feature == 'mask': 
            self.feature_func = lambda x,t : [_convert_example_to_mask_feature(x,t, prompt_idx=0 ), _convert_example_to_mask_feature(x,t, prompt_idx=1 )]
        elif feature == 'all': 
            self.feature_func = lambda x,t : [_convert_example_to_tok_feature(x,t), _convert_example_to_mask_feature(x,t)]
        else:
            raise ValueError( f"feature {feature} is not supported.")


        self.known_feats = cycle([self.feature_func(ex, self.tokenizer) for ex in known_exs])
        self.unknown_feats = cycle([self.feature_func(ex, self.tokenizer) for ex in unknown_exs])

        random.seed(seed)

    def __iter__(self):
        if random.random() < self.mix_ratio:
            yield(next(self.known_feats))
        else:
            yield(next(self.unknown_feats))
        
class MultiviewDataset(Dataset):
    def __init__(self, args, tokenizer: PreTrainedTokenizer, 
        exs: List[InputExample], feature: str='token') -> None:
        super().__init__()
        self.args = args 
        self.tokenizer = tokenizer 

        if feature == 'token': 
            self.feature_func = lambda x, t: [_convert_example_to_tok_feature(x,t ),_convert_example_to_tok_feature(x,t )]
        elif feature == 'mask': 
            self.feature_func = lambda x,t : [_convert_example_to_mask_feature(x,t, prompt_idx=0 ), _convert_example_to_mask_feature(x,t, prompt_idx=1 )]
        elif feature == 'all': 
            self.feature_func = lambda x,t : [_convert_example_to_tok_feature(x,t), _convert_example_to_mask_feature(x,t)]
        else:
            raise ValueError( f"feature {feature} is not supported.")



        self.feats = [self.feature_func(ex, self.tokenizer) for ex in exs]

    def update_pseudo_labels(self, uid2pl: Dict):
        for ex in self.feats:
            for view in ex:
                uid = view['meta']['uid']
                if uid in uid2pl:
                    view['pseudo_label'] = uid2pl[uid]
        return 


    def __len__(self):
        return len(self.feats)

    def __getitem__(self, index):
        return self.feats[index]
      

class MixedBatchMultiviewDataset(Dataset):
    '''
    A dataset that produces a batch with mixed labeled and unlabeled instances.
    '''
    def __init__(self, args, tokenizer: PreTrainedTokenizer, 
        known_exs: List[InputExample], unknown_exs: List[InputExample],  feature: str='token') -> None:
        '''
        Following the UNO paper, we will sample 1 batch from the known classes and 1 batch from the unknown classes. 
        '''
        super().__init__()
        self.args = args 
        self.tokenizer = tokenizer 

        if feature == 'token': 
            self.feature_func = lambda x, t: [_convert_example_to_tok_feature(x,t ), _convert_example_to_tok_feature(x,t )]
        elif feature == 'mask': 
            self.feature_func = lambda x,t : [_convert_example_to_mask_feature(x,t, prompt_idx=0 ), _convert_example_to_mask_feature(x,t, prompt_idx=1 )]
        elif feature == 'all': 
            self.feature_func = lambda x,t : [_convert_example_to_tok_feature(x,t), _convert_example_to_mask_feature(x,t)]
        else:
            raise ValueError( f"feature {feature} is not supported.")

        logger.info('tokenizing features....')

        if args.rev_ratio > 0:
            known_exs = [reverse_ex(ex, args.known_types, args.unknown_types, known=True) if random.random() < args.rev_ratio else ex for ex in known_exs ]
            for ex in unknown_exs:
                ex.label = ex.ori_label+ args.known_types         

        self.known_feats = [self.feature_func(ex, self.tokenizer) for ex in known_exs]
        self.unknown_feats = [self.feature_func(ex, self.tokenizer) for ex in unknown_exs]

    def update_pseudo_labels(self, uid2pl: Dict):
        for ex in self.unknown_feats:
            for view_idx, view in enumerate(ex):
                uid = view['meta']['uid']
                if uid in uid2pl:
                    if isinstance(uid2pl[uid], list):
                        view['pseudo_label'] = uid2pl[uid][view_idx]
                    else:
                        view['pseudo_label'] = uid2pl[uid]
        return 

    def check_pl_acc(self):
        '''
        report pseudo label accuracy
        '''
        labels = []
        pls = []
        for feat in self.unknown_feats:
            labels.append(feat[0]['label'])
            pls.append(feat[0]['pseudo_label'])

        acc = cluster_acc(np.array(labels), np.array(pls), reassign=True)
        return acc 


    def __len__(self):
        return max([len(self.known_feats), len(self.unknown_feats)])

    def __getitem__(self, index):
        labeled_index = int(index % len(self.known_feats))
        labeled_ins = self.known_feats[labeled_index]
        unlabeled_index = int(index % len(self.unknown_feats))
        unlabeled_ins = self.unknown_feats[unlabeled_index]
        return (labeled_ins, unlabeled_ins)


class OpenTypeDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer, dataset_dir ):
        super().__init__()

        self.args = args 
        self.tokenizer = tokenizer 
        self.dataset_name = args.dataset_name # different dataset classes handle 
        self.known_types = args.known_types
        self.dataset_dir = dataset_dir 

    
    def prepare_data(self) -> None:
        return super().prepare_data()

    
    @staticmethod
    def _split_types_tacred(dataset_dir:str, known_types: int)-> Tuple[List[InputExample], List[InputExample]]:
        file_names = ['train.json', 'dev.json','test.json']
        examples = []
        for name in  file_names:
            with open(os.path.join(dataset_dir,name)) as f:
                examples += json.load(f)

        relation_dict = _load_relation_dict(os.path.join(dataset_dir, 'relation_description.csv'))

        example_by_rel = defaultdict(list)
        for ex in examples:
            if ex['relation'] == 'no_relation': continue 
            input_example = _create_example_tacred(ex, relation_dict)
            rel_id = input_example.label 
            example_by_rel[rel_id].append(input_example)
        
        # split by relation
        sorted_rel_ids = sorted(list(example_by_rel.keys()))  # type: List[int]
        known_type_ids = sorted_rel_ids[: known_types]
        unknown_type_ids = sorted_rel_ids[known_types: ]
        known_type_exs = []

        for t in known_type_ids:
            known_type_exs.extend(example_by_rel[t])
        unknown_type_exs = []
        for t in unknown_type_ids:
            # set the known attribute to False 
            for ex in example_by_rel[t]:
                ex.known = False 
            unknown_type_exs.extend(example_by_rel[t])
        
        return known_type_exs, unknown_type_exs


    @staticmethod
    def _split_types_fewrel(dataset_dir: str, known_types: int)-> Tuple[List[InputExample], List[InputExample]]: 
        '''
        For Fewrel the known types are in train_wiki.json and the unknown types are in val_wiki.json.
        '''
        relation_dict = _load_relation_dict(os.path.join(dataset_dir, 'relation_description.csv'), has_qnode=True)
        # known types 
        with open(os.path.join(dataset_dir, 'train_wiki.json'),'r') as f:
            ins_by_rel = json.load(f)

            # unknown types 
        with open(os.path.join(dataset_dir, 'val_wiki.json'),'r') as f:
            val_ins_by_rel = json.load(f)

        ins_by_rel.update(val_ins_by_rel)

        example_by_rel = defaultdict(list)
        for rel_name, ins_list in ins_by_rel.items():
            rel_id = relation_dict.get_id_from_qnode(rel_name) 
            full_rel_name = relation_dict.get_relation(rel_id)
            for idx, instance in enumerate(ins_list):
                input_example = _create_example_fewrel(instance, rel_id, full_rel_name, idx)
                example_by_rel[rel_id].append(input_example)
        
        # split by relation
        sorted_rel_ids = sorted(list(example_by_rel.keys()))  # type: List[int]
        known_type_ids = sorted_rel_ids[: known_types]
        unknown_type_ids = sorted_rel_ids[known_types: ]
        known_type_exs = []
        for t in known_type_ids:
            known_type_exs.extend(example_by_rel[t])
        
        unknown_type_exs = []
        for t in unknown_type_ids:
            # set the known attribute to False 
            for ex in example_by_rel[t]:
                ex.known = False 
            unknown_type_exs.extend(example_by_rel[t])
        
        return known_type_exs, unknown_type_exs

    @staticmethod
    def _data_split_train_test(examples: List[InputExample], test_ratio=0.15)->Tuple[List[InputExample], List[InputExample], List[InputExample]]:
        '''
        split the unknown types into training and test. 
        '''
        total = len(examples)
        random.shuffle(examples)
        train_n = round(total* (1-test_ratio)) 
        train_ex = examples[: train_n]
        test_ex = examples[train_n: ]
        return train_ex, test_ex, test_ex

    @staticmethod 
    def _data_split_train_dev_test(examples: List[InputExample], dev_ratio=0.15, test_ratio=0.15) -> Tuple[List[InputExample], List[InputExample], List[InputExample]]:
        total = len(examples)
        random.shuffle(examples)
        train_n = round(total* (1-test_ratio)) 
        train_ex = examples[: train_n]
        dev_n = round(total * dev_ratio) 
        dev_ex = examples[train_n: train_n + dev_n]
        test_ex = examples[train_n+dev_n: ]
        return train_ex, dev_ex, test_ex 


    @staticmethod
    def collate_batch_feat(batch: List[List[Dict]])-> List[Dict]:
        if isinstance(batch[0], tuple):
            # expand mixed batch 
            new_batch = []
            for tup in batch:
                new_batch.append(tup[0])
                new_batch.append(tup[1])
            batch = new_batch 
        
        assert isinstance(batch[0], list) 

        views_n = len(batch[0])
        output = []
        for i in range(views_n):
            output_i =  {
                'task':'rel',
                'meta': [x[i]['meta'] for x in batch],
                'token_ids': batch_var_length([x[i]['token_ids'] for x in batch]),
                'attn_mask': batch_var_length([x[i]['attn_mask'] for x in batch]),
                'labels': torch.LongTensor([x[i]['label'] for x in batch]),
                'pseudo_labels': torch.LongTensor([x[i]['pseudo_label'] for x in batch]), 
                'known_mask': torch.BoolTensor([x[i]['known'] for x in batch])
            }

            if 'head_span' in batch[0][i]:
                output_i['head_spans'] = torch.LongTensor([x[i]['head_span'] for x in batch])
                output_i['tail_spans'] = torch.LongTensor([x[i]['tail_span'] for x in batch])
            
            if 'mask_bpe_idx' in batch[0][i]:
                output_i['mask_bpe_idx'] = torch.LongTensor([x[i]['mask_bpe_idx'] for x in batch])

            output.append(output_i)

        return output 


    def setup(self, stage: Optional[str]=''): 
        '''
        split the data by known class/unknown class.
        '''
        if self.dataset_name == 'tacred':
            known_type_exs, unknown_type_exs = self._split_types_tacred(self.dataset_dir, self.known_types)    
            logger.info(f'{len(known_type_exs)} known instances, {len(unknown_type_exs)} unknown instances')
            # self.known_train = deepcopy(known_type_exs) # use all training 
            unknown_type_train_exs, unknown_type_test_exs, _ = self._data_split_train_test(unknown_type_exs, test_ratio=self.args.test_ratio)
            known_type_train_exs, known_type_test_exs, _ = self._data_split_train_test(known_type_exs, test_ratio=self.args.test_ratio)
            self.known_train = known_type_train_exs
            self.known_test = known_type_test_exs
            self.unknown_train = unknown_type_train_exs
            self.unknown_test= unknown_type_test_exs
        elif self.dataset_name == 'fewrel':
            known_type_exs, unknown_type_exs = self._split_types_fewrel(self.dataset_dir, self.known_types)
            logger.info(f'{len(known_type_exs)} known instances, {len(unknown_type_exs)} unknown instances')

            unknown_type_train_exs, unknown_type_test_exs, _ = self._data_split_train_test(unknown_type_exs, test_ratio=self.args.test_ratio)
            known_type_train_exs, known_type_test_exs, _ = self._data_split_train_test(known_type_exs, test_ratio=self.args.test_ratio)
            self.known_train = known_type_train_exs
            self.known_test = known_type_test_exs
            self.unknown_train = unknown_type_train_exs
            self.unknown_test= unknown_type_test_exs
        else:
            raise NotImplementedError

    def train_dataloader(self):
        train_dataset = MixedBatchMultiviewDataset(self.args, self.tokenizer, 
            known_exs=self.known_train, 
            unknown_exs=self.unknown_train, 
            feature=self.args.feature
        )
        
        train_dataloader = DataLoader(train_dataset, 
            batch_size = self.args.train_batch_size, 
            shuffle=True, num_workers=self.args.num_workers, 
            pin_memory=True, collate_fn=self.collate_batch_feat) # set to False for extracting features 
        
        return train_dataloader

    def val_dataloader(self):
        unknown_train_dataset = MultiviewDataset(self.args, self.tokenizer, exs=self.unknown_train,  feature=self.args.feature)
        unknown_train_dataloader = DataLoader(unknown_train_dataset, batch_size = self.args.eval_batch_size, 
            shuffle=False, num_workers=self.args.num_workers, 
            pin_memory=True, collate_fn=self.collate_batch_feat)

        unknown_test_dataset = MultiviewDataset(self.args, self.tokenizer, 
            exs=self.unknown_test, 
            feature=self.args.feature)
        unknown_test_dataloader = DataLoader(unknown_test_dataset, batch_size = self.args.eval_batch_size, 
            shuffle=False, num_workers=self.args.num_workers, 
            pin_memory=True, collate_fn=self.collate_batch_feat)


        known_test_dataset= MultiviewDataset(self.args, self.tokenizer, exs=self.known_test,
            feature=self.args.feature)

        known_test_dataloader = DataLoader(known_test_dataset, batch_size = self.args.eval_batch_size, 
            shuffle=False, num_workers=self.args.num_workers, 
            pin_memory=True, collate_fn=self.collate_batch_feat)

        return [unknown_train_dataloader, unknown_test_dataloader, known_test_dataloader]

    def test_dataloader(self):
        unknown_test_dataset = MultiviewDataset(self.args, self.tokenizer, 
            exs=self.unknown_test, 
            feature=self.args.feature)
        unknown_test_dataloader = DataLoader(unknown_test_dataset, batch_size = self.args.eval_batch_size, 
            shuffle=False, num_workers=self.args.num_workers, 
            pin_memory=True, collate_fn=self.collate_batch_feat)

        return unknown_test_dataloader



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='tacred', choices=['tacred', 'fewrel'])
    parser.add_argument('--known_types', type=int , default=31)
    parser.add_argument('--unknown_types', type=int, default=10)
    parser.add_argument('--dataset_dir', type=str, default='data/tacred')
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--feature', type=str, default='all', choices=['token','mask', 'all'])
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0, help='0 for single process')
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--rev_ratio', type=float, default=0.5)

    args = parser.parse_args() 


    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens(['<h>', '</h>', '<t>','</t>'])
    
    data_module = OpenTypeDataModule(args, tokenizer, args.dataset_dir)

    data_module.setup()

    print(len(data_module.known_train))
    print(len(data_module.unknown_train))
    print(len(data_module.unknown_test))


    train_dataset = MixedBatchMultiviewDataset(args, tokenizer, 
            known_exs=data_module.known_train, 
            unknown_exs=data_module.unknown_train, 
            feature=args.feature
            )
    labeled_ins, unlabeled_ins = train_dataset[17]
    tok_input, mask_input = labeled_ins 
    # check the tokens 

    print(tok_input['meta'])
    print(tokenizer.decode(tok_input['token_ids'], skip_special_tokens=True))
    head_span = tok_input['head_span']
    head_bpe = tok_input['token_ids'][head_span[0]: head_span[1]]
    head_entity = tokenizer.decode(head_bpe)
    print(head_entity)


    print(mask_input['meta'])
    print(tokenizer.decode(mask_input['token_ids'], skip_special_tokens=False))
    mask_bpe = mask_input['token_ids'][mask_input['mask_bpe_idx']]
    assert (tokenizer.decode(mask_bpe) == '[MASK]')


    train_dataloader = DataLoader(train_dataset, 
        batch_size = args.train_batch_size, 
        shuffle=True, num_workers=args.num_workers, 
        collate_fn=data_module.collate_batch_feat)


    batch = next(iter(train_dataloader))

    print(batch)
