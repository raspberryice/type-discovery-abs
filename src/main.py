import argparse
import os 
from datetime import datetime
import json

import torch 
import yaml 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from transformers import AutoTokenizer, PreTrainedTokenizer

from data_module import OpenTypeDataModule
from event_data_module import OpenTypeEventDataModule 
from event_e2e_data_module import E2EEventDataModule

from model import TypeDiscoveryModel
from baselines.RoCORE_model import RoCOREModel 
from baselines.etypeclus_model import ETypeClusModel
from baselines.vqvae_model import VQVAEModel 
from baselines.RSN_model import RSNModel 

import common.log as log 
logger = log.get_logger('root')



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_configs', type=str, help='yaml file to load configurations from')
    # Parameters for the model 
    model_arg_group = parser.add_argument_group('model')
    model_arg_group.add_argument('--task', type=str, default='rel', choices=['rel', 'event'])
    model_arg_group.add_argument('--model', type=str, default='tabs', choices=['tabs', 'rocore','etypeclus', 'vqvae','rsn'])
    model_arg_group.add_argument('--collect_features', action='store_true')
    model_arg_group.add_argument('--predict_names', action='store_true')
    model_arg_group.add_argument('--supervised_training', action='store_true', help='train with true labels to check upper bound.')
    model_arg_group.add_argument('--known_types', type=int, default=64, 
        help='the number of types during training. The remaining will be unseen and used for testing.')
    model_arg_group.add_argument('--unknown_types', type=int, default=16)

    model_arg_group.add_argument('--test_ratio', type=float, default=0.15, 
        help='percentage of instances that are used for testing.')
    model_arg_group.add_argument('--incremental', action='store_true', default=False, help='whether or not the test set is mixture of known and unknown.')
    model_arg_group.add_argument('--dataset_name', 
        type=str, 
        default='fewrel', 
        choices=['tacred', 'fewrel', 'ace'])
    

    model_arg_group.add_argument('--feature', type=str, default='all', choices=['token','mask', 'all'])
    model_arg_group.add_argument('--e2e', action='store_true')
    model_arg_group.add_argument('--token_pooling', type=str, default='first', choices=['max','first'])
    model_arg_group.add_argument('--regularization', type=str, default='temp', choices=['sk', 'temp'])
    model_arg_group.add_argument('--temp', type=float, default=0.2, help='value between 0 and 1')
    model_arg_group.add_argument('--sk_epsilon', type=float, default=0.05)
    model_arg_group.add_argument('--psuedo_label', type=str, default='combine', choices=['other','combine','self'])
    model_arg_group.add_argument('--rev_ratio', type=float, default=0.0)
    model_arg_group.add_argument(
        "--model_type", 
        default='bert',
        type=str,
        choices=['bert', 'roberta','albert','gpt2']
    )
    model_arg_group.add_argument('--model_name_or_path', 
        default='bert-base-uncased',
        type=str, 
        help="the model name (e.g., 'roberta-large') or path to a pretrained model")
    model_arg_group.add_argument('--cache_dir', 
        type=str, 
        help='the cache location for transformers library')
    model_arg_group.add_argument('--hidden_dim', type=int, default=256)
    model_arg_group.add_argument('--classifier_layers',type=int, default=2)
 

    model_arg_group.add_argument('--check_pl', action='store_true', help='compute psuedo label accuracy for diagnosis')
    model_arg_group.add_argument('--supervised_pretrain', action='store_true', default=False)
    model_arg_group.add_argument('--label_smoothing_alpha', type=float, default=0.1)
    model_arg_group.add_argument('--label_smoothing_ramp', type=int, default=0)
    model_arg_group.add_argument('--consistency_loss', type=float, default=0.0)
    model_arg_group.add_argument('--pairwise_loss', action='store_true')
    model_arg_group.add_argument('--clustering', type=str, default='kmeans', choices=['online','kmeans','spectral','agglomerative','ward', 'dbscan'])
    model_arg_group.add_argument('--freeze_pretrain', default=False, action='store_true')


    # parameters for RoCORE 
    model_arg_group.add_argument('--center_loss', type=float, default=0.005)
    model_arg_group.add_argument('--sigmoid', type=float, default=2.0)
    model_arg_group.add_argument("--layer", type = int, default = 8)
    model_arg_group.add_argument('--kmeans_dim', type=int, default=256)


    # parameters for etypeclus 
    model_arg_group.add_argument('--temperature', type=float, default=0.1)
    model_arg_group.add_argument('--distribution', default='softmax', choices=['softmax', 'student'])
    model_arg_group.add_argument('--hidden_dims', default='[768, 500, 1000, 100]', type=str)
    model_arg_group.add_argument('--gamma', default=5, type=float, help='weight of clustering loss')

    # parameter for vqvae 
    # note that the gamma parameter is overloaded in this model for the unsupervised loss weight
    model_arg_group.add_argument('--beta', type=float, default=1.0, help='weight for the commitment loss')
    model_arg_group.add_argument('--hybrid', action='store_true', help='whether to use a hybrid vae + vqvae')
    
    # parameters for RSN 
    model_arg_group.add_argument('--perturb_scale', type=float, default=0.02)
    model_arg_group.add_argument('--vat_loss_weight', type=float, default=1.0)
    model_arg_group.add_argument('--use_cnn', action='store_true')
    model_arg_group.add_argument('--p_cond', type=float, default=0.03, help='weight of the conditional ce loss on unknown classes.')
    model_arg_group.add_argument('--clustering_method', type=str, default='spectral', choices=['louvain', 'spectral'], help='clustering method after pairwise metric.')
    # Parameters for IO 
    io_arg_group = parser.add_argument_group('io')
    io_arg_group.add_argument(
        "--ckpt_name",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    io_arg_group.add_argument(
        "--load_ckpt",
        default=None,
        type=str, 
        help='whether to load an existing model. Required for eval.'
    )
    io_arg_group.add_argument('--load_pretrained', type=str, help='load a pretrained model.')
    io_arg_group.add_argument('--dataset_dir', type=str, default='data/fewrel')
  

    # Parameters for runtime 
    runtime_arg_group = parser.add_argument_group('runtime')
    runtime_arg_group.add_argument("--train_batch_size", 
        default=100, type=int, 
        help="Batch size per GPU/CPU for training.")
    runtime_arg_group.add_argument(
        "--eval_batch_size", 
        default=100, type=int, 
        help="Batch size per GPU/CPU for evaluation."
    )
    runtime_arg_group.add_argument(
        "--eval_only", action="store_true",
    )
    runtime_arg_group.add_argument('--num_workers', type=int, default=0)
    runtime_arg_group.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    runtime_arg_group.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    runtime_arg_group.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    runtime_arg_group.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    runtime_arg_group.add_argument("--gradient_clip_val", default=1.0, type=float, help="Max gradient norm.")
    runtime_arg_group.add_argument(
        "--num_train_epochs", default=20, type=int, help="Total number of training epochs to perform."
    )
    runtime_arg_group.add_argument(
        '--num_pretrain_epochs', default=0, type=int
    )
    runtime_arg_group.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    runtime_arg_group.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    
    runtime_arg_group.add_argument("--gpus", default='6,', help='-1 means train on all gpus')
    runtime_arg_group.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    runtime_arg_group.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    
    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed)

    if not args.ckpt_name:
        d = datetime.now() 
        time_str = d.strftime('%m-%dT%H%M')
        args.ckpt_name = '{}_{}_{}_{}'.format(args.model,  args.dataset_name, args.unknown_types,  time_str)


    args.ckpt_dir = os.path.join(f'./checkpoints/{args.ckpt_name}')
    
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # read base arguments from yaml 
    if args.load_configs:
        with open(args.load_configs, 'r') as f:
            yaml_configs = yaml.load(f, Loader=yaml.FullLoader)
        
        for k, v in yaml_configs.items():
            if k=='learning_rate':
                v = eval(v) # convert string to float 
            args.__dict__[k] = v 

    # save the arguments to file 
    arg_dict = vars(args)
    with open(os.path.join(args.ckpt_dir, 'params.json'),'w') as f:
        json.dump(arg_dict, f, indent=2)

    logger.info("Training/evaluation parameters %s", args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        save_top_k=1,
        save_last=True,
        monitor='val/unknown_acc', # metric name 
        mode='max',
        save_weights_only=True,
        filename='{epoch}', # this cannot contain slashes 

    )


    lr_logger = LearningRateMonitor() 
    # TODO: change this to your own project name and username 
    wb_logger = WandbLogger(project='open-type', name=args.ckpt_name, entity='shali0173')


    if args.max_steps < 0 :
        args.max_epochs = args.min_epochs = args.num_train_epochs 
    
    

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)  # type: PreTrainedTokenizer    
    tokenizer.add_tokens(['<h>', '</h>', '<t>','</t>','<tgr>','</tgr>'])
    # tokenizer.add_tokens(['<h>', '</h>', '<t>','</t>']) # quick fix for old checkpoints 
    vocab_size = len(tokenizer)

    if args.task == 'rel':
        dm = OpenTypeDataModule(args, tokenizer, args.dataset_dir)
    elif args.task == 'event':
        if args.e2e:
            dm = E2EEventDataModule(args,tokenizer, args.dataset_dir)
        else:
            dm = OpenTypeEventDataModule(args, tokenizer, args.dataset_dir)
        
    dm.setup()
    train_dm = dm.train_dataloader()
    train_len = len(train_dm) 
    if args.model == 'tabs': 
        model = TypeDiscoveryModel(args, tokenizer, train_len)
    elif args.model == 'rocore':
        model = RoCOREModel(args, tokenizer, train_len)
    elif args.model == 'etypeclus':
        model = ETypeClusModel(args, tokenizer, train_len)
    elif args.model == 'vqvae':
        model = VQVAEModel(args, tokenizer, train_len)
    elif args.model == 'rsn':
        model = RSNModel(args, tokenizer, train_len) 
    else:
        raise ValueError(f"model name {args.model} not recognized.")

    
    trainer = Trainer(
        logger=wb_logger, 
        min_epochs=args.num_train_epochs,
        max_epochs=args.num_train_epochs, 
        gpus=str(args.gpus), # use string or list to specify gpu id  
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val, 
        num_sanity_val_steps=10,
        val_check_interval=1.0, # use float to check every n epochs 
        precision=16 if args.fp16 else 32,
        callbacks = [lr_logger, checkpoint_callback],
        strategy="ddp",
    ) 

    if args.load_ckpt:
        checkpoint = torch.load(args.load_ckpt,map_location=model.device)
        #  enlarge the embeddings based on the state dict 
        for k, v in checkpoint['state_dict'].items():
            if 'embeddings.word_embeddings.weight' in k:
                load_vocab_size = v.size(0)
                
                if load_vocab_size < vocab_size:
                    # this is due to using less special tokens (old checkpoints)
                    # resize the old embedding 
                    embed_dim = v.size(1)
                    new_embeddings = torch.nn.Embedding(vocab_size, embed_dim)
                    new_embeddings.to(model.device)
                    new_embeddings.weight.data[:load_vocab_size, :] = v.data[:load_vocab_size, :]
                    checkpoint['state_dict'][k] = new_embeddings.weight 
                # embeddings = model.pretrained_model.resize_token_embeddings(vocab_size)


        model.load_state_dict(checkpoint['state_dict'], strict=True)  
        model.on_load_checkpoint(checkpoint)
    
    elif args.load_pretrained:
        logger.info(f'loading pretrained model from {args.load_pretrained}')
        checkpoint = torch.load(args.load_pretrained, map_location=model.device)
        model.load_pretrained_model(checkpoint['state_dict'])

    if args.collect_features:
        model.collect_features(train_dm, known=False, raw=False)

    elif args.eval_only: 
        trainer.test(model, datamodule=dm) #also loads training dataloader 
    else:
        # model.initialize_centroids(train_dm)
        trainer.fit(model, datamodule=dm) 
    
if __name__ == "__main__":
    main()