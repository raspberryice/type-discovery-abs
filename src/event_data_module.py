import json
import os 
from collections import defaultdict, Counter  
import random


from typing import List, Dict, Tuple, Callable, Union, Optional
import argparse
from copy import deepcopy
import numpy as np 

from torch.utils.data import DataLoader,  Dataset, ConcatDataset
import torch  
import pytorch_lightning as pl 
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm 


from common.utils import clean_text, cluster_acc
import common.log as log 

logger = log.get_logger('root')

class InputExample(object):
    def __init__(self, unique_id: Union[int, str] , text: List[str], 
        trigger_span: List[int], 
        label: int, event_name: str, known: bool=True):

        self.uid = unique_id
        self.tokens = text # type: List[str]
        self.trigger_span = trigger_span 
        self.ori_label = label 
        self.label = label
        self.known = known 
        self.event_name = event_name 
        self.pseudo_label = -1
    def __str__(self) -> str:
        return str(self.uid)




class EventDict(object):
    def __init__(self) -> None:
        self.event2int = {}
        self.int2event = {}

    def add(self, event: str) -> None:
        evt_id = len(self.event2int)
        self.event2int[event] = evt_id
        self.int2event[evt_id] = event 
        return 
    
    def get_id(self, event: str) -> int:
        return self.event2int[event]

    def get_event(self, id:int) -> str:
        return self.int2event[id]


    def __len__(self):
        return len(self.event2int)
    
     


def _load_event_dict(event_file_path: str) -> EventDict:
    event_dict = EventDict() 

    with open(event_file_path, 'r') as f:
        event_file = json.load(f)
    
    for k in event_file:
        event_dict.add(k)
    return event_dict 

def _create_event_dict_from_list(events:list) -> EventDict:
    event_dict = EventDict()

    for event in events:
        event_dict.add(event)
    return event_dict 


def _create_example_for_sent_ace(instance:Dict, max_word_len: int=200) -> List[InputExample]:
    text = instance['tokens']

    sent_exs = []
    for event in instance['event_mentions']:
        uid = event['id'] 
        label = -1 # placeholder 
        trigger_span = [event['trigger']['start'], event['trigger']['end']] 
        if len(text) > max_word_len:
            trunc_text = deepcopy(text)
            while len(trunc_text) > max_word_len:
                if trigger_span[0] > len(trunc_text) - trigger_span[1]:
                    trunc_text = trunc_text[1:]
                else:
                    trunc_text = trunc_text[:-1]
    
        input_example = InputExample(uid, text, trigger_span, label, event['event_type'])
        sent_exs.append(input_example)
    return sent_exs

def _create_example_for_sent_maven(instance: Dict, event_type: str, event_dict:EventDict, max_word_len: int=200) -> List[InputExample]:
    '''
    One sentence may contain multiple events. Remove the instances that do not match with the event type 
    '''
    text = instance['tokens']

    sent_exs = []
    for idx, event in enumerate(instance['events']):
        if event[0] != event_type: continue 
        uid = f"{instance['sid']}_{idx}"
        label = event_dict.get_id(event_type) 
        trigger_span = [event[1]['start'], event[1]['end']] 
        if len(text) > max_word_len:
            trunc_text = deepcopy(text)
            while len(trunc_text) > max_word_len:
                if trigger_span[0] > len(trunc_text) - trigger_span[1]:
                    trunc_text = trunc_text[1:]
                else:
                    trunc_text = trunc_text[:-1]
    
        input_example = InputExample(uid, text, trigger_span, label, event_type)
        sent_exs.append(input_example)
    return sent_exs



def _convert_example_to_tok_feature(ex: InputExample, tokenizer:PreTrainedTokenizer) -> Dict:
    '''
    replace the tgr with mask if needed.
    '''
    tokens = ex.tokens 
    trigger_tokens = tokens[ex.trigger_span[0]: ex.trigger_span[1] ]
  
    meta = {
        'uid': ex.uid,
        'tokens': ex.tokens,
        'known': ex.known,
        'label': ex.event_name,
        'feature_type': 'token',
        'trigger': ' '.join(trigger_tokens)
    }

    # insert tgr markers 
    input_tokens = deepcopy(tokens)
    input_tokens.insert(ex.trigger_span[0], '<tgr>')
    input_tokens.insert(ex.trigger_span[1]+1, '</tgr>')

    # # without tgr 
    # substart = tokenizer.encode(' '.join(input_tokens[:ex.trigger_span[0]]))
    # subend = tokenizer.encode(' '.join(input_tokens[:ex.trigger_span[1]]))
    # trigger_bpe_span = (len(substart) -1, len(subend) -1)
    
    
    # get the head span and tail span in bpe offset 
    # this is with the tgr marker 
    substart = tokenizer.encode(' '.join(input_tokens[:ex.trigger_span[0] +1])) 
    subend = tokenizer.encode(' '.join(input_tokens[:ex.trigger_span[1]+1]))
    trigger_bpe_span = (len(substart) -1, len(subend) -1)
  

    sentence = ' '.join(input_tokens)

    token_ids = tokenizer.encode(sentence, return_tensors='pt').squeeze(0) # (seq_len) 
    seq_len = token_ids.size(0)
    attn_mask = torch.ones((seq_len))

    return {
        'meta': meta,
        'token_ids': token_ids,
        'attn_mask': attn_mask,
        'trigger_span': trigger_bpe_span, 
        'label': ex.label,
        'known': ex.known,
        'pseudo_label': ex.pseudo_label
    }

def _convert_example_to_mask_feature(ex: InputExample, tokenizer: PreTrainedTokenizer, prompt_idx: int =0) -> Dict: 
    tokens = ex.tokens
    trigger_tokens = tokens[ex.trigger_span[0]: ex.trigger_span[1] ]
  
    meta = {
        'uid': ex.uid,
        'tokens': ex.tokens,
        'known': ex.known,
        'label': ex.event_name,
        'feature_type': 'mask',
        'trigger': ' '.join(trigger_tokens)
    }

    input_tokens = deepcopy(tokens)
    input_tokens.insert(ex.trigger_span[0], '<tgr>')
    input_tokens.insert(ex.trigger_span[1]+1, '</tgr>')

    if prompt_idx == 0:
        # trigger is a <mask> event 
        prompt = trigger_tokens + ['is','a', tokenizer.mask_token, 'event']  
        mask_word_prefix = input_tokens + trigger_tokens  + ['is', 'a']
    elif prompt_idx ==1: 
        prompt = trigger_tokens + ['refers','to', 'a', tokenizer.mask_token, 'event']  
        mask_word_prefix = input_tokens + trigger_tokens  + ['refers','to', 'a']
    else:
        raise ValueError(f'prompt {prompt_idx} not defined')

    prefix_bpe = tokenizer.encode(' '.join(mask_word_prefix))
    mask_bpe_idx = len(prefix_bpe) -1 # remove the [sep] token at the end 

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


def batch_var_length(tensors: List[torch.Tensor], max_length_thres: int =342):
    '''
    max_length_thres was selected using the MAVEN dataset. 
    '''
    batch_size = len(tensors)
    max_len =  max([t.size(0) for t in tensors])
    if max_len > max_length_thres:
        logger.info(f'encountering seq len of {max_len}, this might be problematic')
    pad_len = min(max_length_thres,max_len) 
    batch_tensors = torch.zeros((batch_size, pad_len)).type_as(tensors[0])
    for i in range(batch_size):
        actual_len = min(pad_len, tensors[i].size(0))
        batch_tensors[i, :actual_len] = tensors[i][:actual_len]
    
    return batch_tensors 


        
class MultiviewDataset(Dataset):
    def __init__(self, args, tokenizer: PreTrainedTokenizer, 
        exs: List[InputExample], feature: str='token') -> None:
        super().__init__()
        self.args = args 
        self.tokenizer = tokenizer 

        if feature == 'token': 
            self.feature_func = lambda x, t: [_convert_example_to_tok_feature(x,t ), _convert_example_to_tok_feature(x,t)]
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
            self.feature_func = lambda x, t: [_convert_example_to_tok_feature(x,t ), _convert_example_to_tok_feature(x,t)]
        elif feature == 'mask': 
            self.feature_func = lambda x,t : [_convert_example_to_mask_feature(x,t, prompt_idx=0 ), _convert_example_to_mask_feature(x,t, prompt_idx=1 )]
        elif feature == 'all': 
            self.feature_func = lambda x,t : [_convert_example_to_tok_feature(x,t), _convert_example_to_mask_feature(x,t)]
        else:
            raise ValueError( f"feature {feature} is not supported.")

        logger.info('tokenizing features....')
     

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


class OpenTypeEventDataModule(pl.LightningDataModule):
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
    def _split_types_maven(dataset_dir: str, known_types:int) -> Tuple[List[InputExample], List[InputExample]]:
        '''
        :return known_type_exs, unknown_type_exs 
        '''
        file_names = ['maven_fewshot_train.json', 'maven_fewshot_dev.json', 'maven_fewshot_test.json']
        examples = {} # event type -> List of instances 
        event_dict = _load_event_dict(os.path.join(dataset_dir,'ontology.json'))

        for name in file_names:
            with open(os.path.join(dataset_dir, name)) as f:
                new_exs = json.load(f)
                for k, v in new_exs.items():
                    converted_exs = [ex for s in v for ex in _create_example_for_sent_maven(s, k, event_dict)] 
                    if k not in examples:
                        examples[k] = converted_exs
                    else:
                        examples[k].extend(converted_exs)
       
        assert (len(event_dict) == len(examples))
        example_by_id = {event_dict.get_id(k): v for k,v in examples.items()} 
        sorted_ids = sorted(list(example_by_id.keys()))
        known_type_ids = sorted_ids[: known_types]
        unknown_type_ids = sorted_ids[known_types: ]
        known_type_exs = []
        for t in known_type_ids: known_type_exs.extend(example_by_id[t])
        unknown_type_exs = []
        for t in unknown_type_ids: 
            for ex in example_by_id[t]:
                ex.known = False 
            unknown_type_exs.extend(example_by_id[t])
        return known_type_exs, unknown_type_exs

    
    @staticmethod 
    def _split_types_ace(dataset_dir: str, known_types: int) -> Tuple[List[InputExample], List[InputExample]]:
        #TODO: change the path of ACE input files if needed 
        file_names = ['pro_mttrig_id/json/train.oneie.json','pro_mttrig_id/json/dev.oneie.json','pro_mttrig_id/json/test.oneie.json']
     

        example_by_name = defaultdict(list)

        for name in file_names:
            with open(os.path.join(dataset_dir, name)) as f:
                for line in f:
                    sent = json.loads(line)
                    converted_exs =  _create_example_for_sent_ace(sent) # List [InputExamples]
                    for ex in converted_exs: 
                        name = ex.event_name 
                        example_by_name[name].append(ex)
        
        sorted_names = sorted(list(example_by_name.keys()), key= lambda x: len(example_by_name[x]), reverse=True)
        event_dict = _create_event_dict_from_list(sorted_names) 
        example_by_id = {event_dict.get_id(k): v  for k,v in example_by_name.items()} 
        for k, exs in example_by_id.items():
            for ex in exs:
                ex.label = k 
        
        sorted_ids = sorted(list(example_by_id.keys()))
        known_type_ids = sorted_ids[: known_types]
        unknown_type_ids = sorted_ids[known_types: ]
        known_type_exs = []
        for t in known_type_ids: known_type_exs.extend(example_by_id[t])
        unknown_type_exs = []
        for t in unknown_type_ids: 
            for ex in example_by_id[t]:
                ex.known = False 
            unknown_type_exs.extend(example_by_id[t])
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
                'task': 'event',
                'meta': [x[i]['meta'] for x in batch],
                'token_ids': batch_var_length([x[i]['token_ids'] for x in batch]),
                'attn_mask': batch_var_length([x[i]['attn_mask'] for x in batch]),
                'labels': torch.LongTensor([x[i]['label'] for x in batch]),
                'pseudo_labels': torch.LongTensor([x[i]['pseudo_label'] for x in batch]), 
                'known_mask': torch.BoolTensor([x[i]['known'] for x in batch])
            }

            if 'trigger_span' in batch[0][i]:
                output_i['trigger_spans'] = torch.LongTensor([x[i]['trigger_span'] for x in batch])
            
            if 'mask_bpe_idx' in batch[0][i]:
                output_i['mask_bpe_idx'] = torch.LongTensor([x[i]['mask_bpe_idx'] for x in batch])

            output.append(output_i)

        return output 


    def setup(self, stage: Optional[str]=''): 
        '''
        split the data by known class/unknown class.
        '''
        if self.dataset_name == 'maven':
            known_type_exs, unknown_type_exs = self._split_types_maven(self.dataset_dir, self.known_types)
        elif self.dataset_name =='ace':
            known_type_exs, unknown_type_exs = self._split_types_ace(self.dataset_dir, self.known_types)
        else:
            raise ValueError(f'{self.dataset_name} not supported')
        
        logger.info(f'{len(known_type_exs)} known instances, {len(unknown_type_exs)} unknown instances')
        unknown_type_train_exs, unknown_type_test_exs, _ = self._data_split_train_test(unknown_type_exs, test_ratio=self.args.test_ratio)
        known_type_train_exs, known_type_test_exs, _ = self._data_split_train_test(known_type_exs, test_ratio=self.args.test_ratio)
        self.known_train = known_type_train_exs
        self.known_test = known_type_test_exs
        self.unknown_train = unknown_type_train_exs
        self.unknown_test= unknown_type_test_exs
    
    def train_dataloader(self):
        train_dataset = MixedBatchMultiviewDataset(self.args, self.tokenizer, 
            known_exs=self.known_train, 
            unknown_exs=self.unknown_train, 
            feature=self.args.feature
        )
        
        train_dataloader = DataLoader(train_dataset, 
            batch_size = self.args.train_batch_size, 
            shuffle=True, num_workers=self.args.num_workers, 
            pin_memory=False, collate_fn=self.collate_batch_feat) # set to False for extracting features 
        
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
        # unknown_train_dataset = SimpleDataset(self.args, self.tokenizer, exs=self.unknown_train, feature=self.args.feature)
        # unknown_train_dataloader = DataLoader(unknown_train_dataset, batch_size = self.args.eval_batch_size, 
        #     shuffle=False, num_workers=self.args.num_workers, 
        #     pin_memory=True, collate_fn=self.collate_batch_feat)

        unknown_test_dataset = MultiviewDataset(self.args, self.tokenizer, 
            exs=self.unknown_test, 
            feature=self.args.feature)
           
        if self.args.incremental:
            known_test_dataset= MultiviewDataset(self.args, self.tokenizer, exs=self.known_test,
            feature=self.args.feature)
            test_dataloader = DataLoader(ConcatDataset([known_test_dataset, unknown_test_dataset]), batch_size = self.args.eval_batch_size, 
                shuffle=False, num_workers=self.args.num_workers, 
                pin_memory=True, collate_fn=self.collate_batch_feat)
            return test_dataloader 
        
        else:
            unknown_test_dataloader = DataLoader(unknown_test_dataset, batch_size = self.args.eval_batch_size, 
            shuffle=False, num_workers=self.args.num_workers, 
            pin_memory=True, collate_fn=self.collate_batch_feat)

            return unknown_test_dataloader



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ace', choices=['maven','ace'])
    parser.add_argument('--known_types', type=int , default=10)
    parser.add_argument('--unknown_types', type=int, default=23)
    parser.add_argument('--dataset_dir', type=str, default='data/ace')
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--feature', type=str, default='all', choices=['token','mask', 'all'])
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0, help='0 for single process')
    parser.add_argument('--eval_batch_size', type=int, default=4)

    args = parser.parse_args() 


    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens(['<tgr>', '</tgr>'])
    
    data_module = OpenTypeEventDataModule(args, tokenizer, args.dataset_dir)

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
    head_span = tok_input['trigger_span']
    head_bpe = tok_input['token_ids'][head_span[0]: head_span[1]]
    trigger_word = tokenizer.decode(head_bpe)
    print(trigger_word)


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
