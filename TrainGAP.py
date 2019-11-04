#!pip install transformers

import pandas as pd
import numpy as np
import transformers
from transformers import BertPreTrainedModel, BertTokenizer, BertModel, BertConfig
import torch
from torch.utils.data import DataLoader,TensorDataset, RandomSampler, SequentialSampler, DistributedSampler
import math
from tqdm import trange
from tqdm import tqdm as tqdm
import torch.distributed as dist
import torch.multiprocessing as mp

import sys,logging
import os,time
import tempfile

import dist_util
from dataset import get_features_from_example,create_dataset
from BertModels import *
from arguments import parser

logging.root.handlers = []
logging.basicConfig(level="INFO", 
                    format = '%(asctime)s:%(levelname)s: %(message)s' ,
                    stream = sys.stdout)
logger = logging.getLogger(__name__)
logger.info('hello')
MODEL_CLASSES = {'concat' : BertForPronounResolution_Concat,
                'mul' : BertForPronounResolution_Mul,
                'segment' : BertForPronounResolution_Segment
                }

def init_logger(local_rank):
    global logger 
    logging.root.handlers = []
    logging.basicConfig(level="INFO", 
                    format = 'Ranks {}: %(asctime)s:%(levelname)s: %(message)s'.format(local_rank) ,
                    stream = sys.stdout)
    logger = logging.getLogger(__name__)

def initialize(args):
    
    # Create folders
    args.cache_dir = os.path.join(tempfile.gettempdir(),str(args.local_rank))
    os.makedirs(args.cache_dir,exist_ok = True)

    # Create datasets

    train_df =  pd.read_pickle(os.path.join(args.input_dir,'train_processed.pkl'))
    
    if args.sample_limit:
        train_df = train_df.iloc[:args.sample_limit]
    
    val_df =  pd.read_pickle(os.path.join(args.input_dir,'val_processed.pkl'))

    train_dataset = create_dataset(train_df,args.cache_dir, args.bert_type)
    val_dataset = create_dataset(val_df,args.cache_dir, args.bert_type)

    #Create model

    model = MODEL_CLASSES[args.model_type].from_pretrained(args.bert_type, cache_dir = args.cache_dir)
    if type(model) is BertForPronounResolution_Segment:
        model.post_init()
    
    logger.info(f'Model used = {type(model)}')
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr) #change it to AdamW later

    #fp16 AMP changes
    if args.fp16:
        from apex import amp
        # This needs to be done before wrapping with DDP or horovod.
        torch.cuda.set_device(args.device) #not sure if it's required
        amp.initialize(model,optimizer,args.amp_opt_level)

    if args.isaml:
        from azureml.core import Run
        args.run = Run.get_context()

    return model, optimizer, train_dataset, val_dataset


def evaluate(val_dataloader,model, args):
    torch.cuda.empty_cache() # 7 GB of cached memory seen
    
    all_labels = []
    all_preds = []
    total_loss = 0
    acc = 0
    model.eval()
    steps = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            loss,logits = model(*batch)
            total_loss+=loss.item()
            

            #converting to float to avoid this error
            #"argmax_cuda" not implemented for 'Half'
            preds = torch.argmax(logits.float(), dim = 1)
            labels = batch[-1]
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            steps += 1


    acc = np.sum(np.array(all_preds) == np.array(all_labels))/ len(all_preds)
    loss = total_loss/steps
    return loss,acc

import time
def train(train_dataloader, val_dataloader, model, optimizer, args ):

    losses = []
    total_loss = 0
    
    for epoch in range(args.epochs):
        start = time.time()
        logger.info(f'Training epoch {epoch}')
        model.train()
        batch_iterator = tqdm(train_dataloader, desc='batch_iterator')
        
        for step, batch in enumerate(batch_iterator):
            batch = (t.to(args.device) for t in batch)

            # disable require_forward_param_sync 
            #start = time.time()
            loss,logits = model(*batch)
            #end = time.time()
            #logger.info(f'forward :{step} time = {end-start}')            
            
            #start = time.time()
            
            # Disabling reduction.Does not work. Check DDP source code later
            #if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            #    model.require_backward_grad_sync = True \
            #        if (step+1) % args.gradient_accumulation == 0 else False
                
            loss = loss/args.gradient_accumulation
            if args.fp16:
                from apex import amp
                with amp.scale_loss(loss,optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            
            # Does this mean horovod reduction which happens at optimizer.step() is FP32?

            total_loss+=loss.item()

            #end = time.time()
            #add context manager instead
            #logger.info(f'backward :{step} time = {end-start}')
            
            if (step+1) % args.gradient_accumulation == 0:
                optimizer.step()

                optimizer.zero_grad()
                losses.append(total_loss)
                log_aml(args,'train_loss', total_loss)
                batch_iterator.set_postfix({'loss':losses[-1]}, refresh=True)
                total_loss=0

        end = time.time()
        logger.info(f'Time taken for epoch {epoch+1} is = {end-start}')
        log_aml(args, 'epoch_time', end-start)
        val_loss , val_acc = evaluate(val_dataloader, model, args)
        logger.info(f'Epoch = {epoch+1}, Val loss = {val_loss}, val_acc = {val_acc}')
        log_aml(args, 'val_loss', val_loss)
        log_aml(args, 'val_acc', val_acc)
    return losses, val_loss , val_acc

def log_aml(args, key,val):
    if args.isaml:
        if not args.is_distributed or args.global_rank == 0:
            args.run.log(key,val)

def finish(args, model, optimizer):
    if args.local_rank == -1 or (args.is_distributed and args.global_rank == 0):
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(),'model_checkpoint.pt')
        if args.fp16:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict()
            }
            torch.save(checkpoint, 'amp_checkpoint.pt')


def main(args):
    logger.info('Starting single GPU/CPU training')   
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')
    args.device = device
    
    
    logger.info (args)

    model, optimizer, train_dataset, val_dataset = initialize(args)
    
    sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,batch_size= args.batch_size, sampler = sampler)

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,batch_size= args.val_batch_size, sampler = val_sampler)

    metrics = train(train_dataloader, val_dataloader, model, optimizer, args)
    finish(args, model, optimizer)


def distributed_main(args):
    '''
    Similar to main but sets the mode and data loader for distributed programming.
    '''


    args.device = torch.device(args.local_rank) # <--
    init_logger(args.local_rank)
    
    logger.info (args)

    dist_util.ddp_setup(args)
    
    model, optimizer, train_dataset, val_dataset = initialize(args)

    #Understand more about find_unused_parameters later. This is required.
    model = torch.nn.parallel.DistributedDataParallel(model, 
                                                        device_ids=[args.device],
                                                        output_device=args.device,
                                                        find_unused_parameters = True) # <--

    sampler = DistributedSampler(train_dataset, num_replicas = args.world_size)

    train_dataloader = DataLoader(train_dataset,batch_size= args.batch_size, sampler = sampler)# <--

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,batch_size= args.val_batch_size, sampler = val_sampler)

    metrics = train(train_dataloader, val_dataloader, model, optimizer, args)
    finish(args, model, optimizer)

def distributed_main_horovod(args):
    import horovod.torch as hvd
    #hvd.init()
    
    print('hvd.local_rank()',hvd.local_rank())
    args.device = torch.device(hvd.local_rank())# <--
    init_logger(hvd.local_rank())
    
    logger.info('Using horovod for distributed training')

    model, optimizer, train_dataset, val_dataset = initialize(args)
    # Broadcast parameters from rank 0 to all other processes.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    train_dataloader = DataLoader(train_dataset,batch_size= args.batch_size, sampler = sampler)# <--
    
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,batch_size= args.val_batch_size, sampler = val_sampler)

    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters = model.named_parameters(), backward_passes_per_step = args.gradient_accumulation)# <--
    
    metrics = train(train_dataloader, val_dataloader, model, optimizer, args)
    finish(args, model, optimizer)

def recalculate_ga(args):
    args.batch_size =  4 if args.fp16 else 2 #Bert Large
    if 'bert-base' in args.bert_type:
        args.batch_size *= 2 #Bert Base
    args.gradient_accumulation = args.per_gpu_batch_size//args.batch_size

if __name__ == '__main__':
    args = parser.parse_args()
    recalculate_ga(args)
    
    logger.info (args)

    if args.is_distributed:
        dist_util.run_distributed(args)
    else:
        main(args)
