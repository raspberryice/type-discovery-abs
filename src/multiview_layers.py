from typing import List, Dict, Tuple, Optional 

import torch 
from torch import nn 
import torch.nn.functional as F 
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

from common.layers import ClassifierHead, MLP, CommonSpaceCache, ReconstructionNet
from common.center_loss import CenterLoss 
from common.utils import onedim_gather



import common.log as log
logger = log.get_logger('root')

class MultiviewModel(nn.Module):
    def __init__(self, args, model_config, pretrained_model, unfreeze_layers: List=[]) -> None:
        super().__init__()
        self.args = args 
        
        self.layer = args.layer 
        if args.freeze_pretrain:
            self.pretrained_model = self.finetune(pretrained_model, unfreeze_layers)  
        else:
            self.pretrained_model = pretrained_model 

        self.views = nn.ModuleList() 
        if args.feature == 'all': feature_types = ['token','mask']
        elif args.feature == 'mask': feature_types = ['mask', 'mask']
        elif args.feature == 'token': feature_types = ['token', 'token']
        else: 
            raise NotImplementedError

        if self.args.rev_ratio >0:
            known_head_types = 2 * args.known_types 
        else:
            known_head_types = args.known_types 
        
        for view_idx, ft in enumerate(feature_types):
            if ft == 'mask':
                view_model = nn.ModuleDict(
                    {
                        'common_space_proj': MLP(model_config.hidden_size, args.hidden_dim, args.kmeans_dim,
                    norm=True, norm_type='batch', layers_n=2, dropout_p =0.1),
                        'known_type_center_loss': CenterLoss(args.kmeans_dim, args.known_types, weight_by_prob=False),
                        'unknown_type_center_loss': CenterLoss(args.kmeans_dim, args.unknown_types, weight_by_prob=False),
                        'known_type_classifier': ClassifierHead(args, args.kmeans_dim, 
                    known_head_types, layers_n=args.classifier_layers, n_heads=1, dropout_p=0.0, hidden_size=args.kmeans_dim),
                        'unknown_type_classifier': ClassifierHead(args, args.kmeans_dim, 
                    args.unknown_types, layers_n=args.classifier_layers, n_heads=1, dropout_p=0.0, hidden_size=args.kmeans_dim)
                    }     
                )
            else:
                if self.args.task == 'rel': 
                    input_size = 2 * model_config.hidden_size # head, tail
                else:
                    input_size = model_config.hidden_size # trigger 
                
                view_model = nn.ModuleDict(
                    {
                        'common_space_proj': MLP(input_size, args.hidden_dim, args.kmeans_dim,
                    norm=True, norm_type='batch', layers_n=2, dropout_p=0.1),
                        'known_type_center_loss': CenterLoss(args.kmeans_dim, args.known_types, weight_by_prob=False),
                        'unknown_type_center_loss': CenterLoss(args.kmeans_dim, args.unknown_types, weight_by_prob=False),
                        'known_type_classifier': ClassifierHead(args, args.kmeans_dim, 
                    known_head_types, layers_n=args.classifier_layers, n_heads=1, dropout_p=0.0, hidden_size=args.kmeans_dim),
                        'unknown_type_classifier': ClassifierHead(args, args.kmeans_dim, 
                    args.unknown_types, layers_n=args.classifier_layers, n_heads=1, dropout_p=0.0, hidden_size=args.kmeans_dim)
                    }
                )
            self.views.append(view_model)

        # this commonspace means that known classes and unknown classes are projected into the same space 
        self.commonspace_cache = nn.ModuleList([
            CommonSpaceCache(feature_size=args.kmeans_dim, known_cache_size=512, unknown_cache_size=256, sim_thres=0.8),
            CommonSpaceCache(feature_size=args.kmeans_dim, known_cache_size=512, unknown_cache_size=256, sim_thres=0.8)
        ])

        return 
    

    # FIXME: this function is taken from ROCORE, will use layer.7 instead of layer.8 as described in the paper. 
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


    def generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate the default inputs required by almost every language model."""
        inputs = {'input_ids': batch['token_ids'], 'attention_mask': batch['attn_mask']}
        return inputs

    def predict_name(self, batch:Dict, topk: int=10) -> torch.LongTensor:
        lm_inputs = self.generate_default_inputs(batch)
        outputs = self.pretrained_model(**lm_inputs)
        vocab_logits = outputs[0] #(batch_size, seq, vocab_size)
        mask_bpe_idx = batch['mask_bpe_idx']
        mask_logits = onedim_gather(vocab_logits, dim=1, index=mask_bpe_idx.unsqueeze(1)).squeeze(1) # (batch_size, vocab_size)
        predicted_token_ids = mask_logits.argsort(dim=-1, descending=True)[:, :topk]
        return predicted_token_ids
        

    def _compute_features(self, batch:Dict, method: str='token', pooling: str='first')-> torch.FloatTensor:
        lm_inputs = self.generate_default_inputs(batch)
        outputs = self.pretrained_model(**lm_inputs)
        # seq_output = outputs[0] # last layer output, only works with AutoModel, does not work with AutoModelForMaskedLM
        
        all_encoder_layers = outputs[-1]
        seq_output  = all_encoder_layers[-1]
    
        if method == 'token': 
            if batch['task'] == 'rel':
                head_spans = batch['head_spans'] # (batch, 2)
                tail_spans = batch['tail_spans'] # (batch,2 )
                batch_size = head_spans.size(0)

                if pooling == 'first':
                    # taking the first token as the representation for the entity
                    head_rep = onedim_gather(seq_output, dim=1, index=head_spans[:, 0].unsqueeze(1))
                    tail_rep = onedim_gather(seq_output, dim=1, index=tail_spans[:, 0].unsqueeze(1))
                    feat = torch.cat([head_rep, tail_rep], dim = 2).squeeze(1) 
                elif pooling == 'max':
                    # max pooling over the tokens in the entity 
                    head_rep_list = []
                    tail_rep_list = []
                    for i in range(batch_size):
                        head_ent = seq_output[i, head_spans[i, 0]: head_spans[i, 1], :] #(ent_len, hidden_dim)
                        head_ent_max, _ = torch.max(head_ent, dim=0)
                        head_rep_list.append(head_ent_max)

                        tail_ent = seq_output[i, tail_spans[i, 0]: tail_spans[i, 1], :] #(ent_len, hidden_dim)
                        tail_ent_max, _ = torch.max(tail_ent, dim=0)
                        tail_rep_list.append(tail_ent_max)
                    head_rep = torch.stack(head_rep_list, dim=0)
                    tail_rep = torch.stack(tail_rep_list, dim=0)
                    feat= torch.cat([head_rep, tail_rep], dim=1)
            elif batch['task'] == 'event':
                trigger_spans = batch['trigger_spans']
                batch_size = trigger_spans.size(0)
                if pooling == 'first':
                    feat = onedim_gather(seq_output, dim=1, index=trigger_spans[:, 0].unsqueeze(1)).squeeze(1)
                elif pooling == 'max':
                    rep_list = []
                    for i in range(batch_size):
                        tgr = seq_output[i, trigger_spans[i, 0]: trigger_spans[i, 1], :]
                        tgr_max, _ = torch.max(tgr, dim=0)
                        rep_list.append(tgr_max)
                    feat = torch.stack(rep_list, dim=0)

        elif method == 'mask':
            mask_bpe_idx = batch['mask_bpe_idx'] # (batch)
            seq_len = seq_output.size(1)
            assert (mask_bpe_idx.max() < seq_len), "mask token out of bounds"
            feat = onedim_gather(seq_output, dim=1, index=mask_bpe_idx.unsqueeze(1)).squeeze(1)
            
        else:
            raise NotImplementedError
        return feat 

    
    def _compute_prediction_logits(self, batch: Dict,
             method:str ='token', pooling: str ='first',
             view_idx: int =0)-> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        '''
        :param seq_output: (batch, seq_len, hidden_dim)
        :param method: str, one of 'token', 'mask'
        :param pooling: str, one of first, max 
        :param view_idx: int 
        '''
        feat = self._compute_features(batch, method, pooling)

        view_model = self.views[view_idx]
        common_space_feat = view_model['common_space_proj'](feat)
        known_head_logits = view_model['known_type_classifier'](common_space_feat)
        unknown_head_logits = view_model['unknown_type_classifier'](common_space_feat)
       
        predicted_logits = torch.cat([known_head_logits, unknown_head_logits], dim=1) 

        
        return predicted_logits, common_space_feat, feat

    def _on_train_batch_start(self):
         # normalize all centroids 
        for view_model in self.views: 
            view_model['known_type_classifier'].update_centroid() 
            view_model['unknown_type_classifier'].update_centroid() 

        return 
    

    def update_centers(self, centers: torch.FloatTensor, known:bool=True, view_idx: int=0): 
        if known:
            self.views[view_idx]['known_type_center_loss'].centers = centers 
        else:
            self.views[view_idx]['unknown_type_center_loss'].centers = centers 
        return 
