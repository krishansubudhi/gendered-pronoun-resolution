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
from BertModels import *
from arguments import parser
import torch.distributed as dist
import torch.multiprocessing as mp

import sys,logging
import os,time
import dist_util

logging.root.handlers = []
logging.basicConfig(level="INFO", 
                    format = '%(asctime)s:%(levelname)s: %(message)s' ,
                    stream = sys.stdout)
logger = logging.getLogger(__name__)
logger.info('hello')

def get_features_from_example(ex, tokenizer):
    cls_id = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    sep_id = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
    max_length = 512

    input = ex.input.copy()
    pab = ex.pab_pos.copy()

    #add special tokens [CLS at beginning], [SEP at end], [optional SEP before pos]
    input = [cls_id]+tokenizer.convert_tokens_to_ids(input.tolist())+[sep_id]
    pab += 1
    
    #attention masking and padding
    mask = [1] * len(input)
    pad_length = max_length -len(input)
    #padding tokens and mask with 0
    input = input + [0]*pad_length
    mask = mask + [0]*pad_length

    assert len(input) == max_length
    assert len(mask) == max_length
    
    return input, mask, pab, int(ex.label)

def create_dataset(df):
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    features = [get_features_from_example(df.iloc[i],tokenizer) for i in range(len(df))]

    ids = torch.tensor([feature[0] for feature in features])
    masks = torch.tensor([feature[1] for feature in features])
    pabs = torch.tensor([feature[2] for feature in features])
    labels = torch.tensor([feature[3] for feature in features])

    #logger.info(ids.size(), masks.size(), pabs.size(), labels.size())

    return TensorDataset(ids, masks, pabs, labels)



def evaluate(val_dataloader,model, args):
    all_labels = []
    all_preds = []
    total_loss = 0
    acc = 0
    model.eval()
    steps = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            labels = batch[-1]
            batch = tuple(t.to(args.device) for t in batch)
            loss,logits = model(*batch)
            preds = torch.argmax(logits, dim = 1)

            total_loss+=loss.item()
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            steps += 1


    acc = np.sum(np.array(all_preds) == np.array(all_labels))/ len(all_preds)
    loss = total_loss/steps
    return loss,acc

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
            loss,logits = model(*batch)

            #logger.info(f'step = {step}, loss = {losses[-1]}')

            loss = loss/args.gradient_accumulation
            loss.backward()

            total_loss+=loss.item()

            if (step+1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                losses.append(total_loss)
                batch_iterator.set_postfix({'loss':losses[-1]}, refresh=True)
                #batch_iterator.write(f'step = {step}, loss = {total_loss}')
                total_loss=0
        end = time.time()
        logger.info(f'Time taken for epoch {epoch+1} is = {end-start}')
        val_loss , val_acc = evaluate(val_dataloader, model, args)
        logger.info(f'Epoch = {epoch+1}, Val loss = {val_loss}, val_acc = {val_acc}')
        
    return losses, val_loss , val_acc

    
MODEL_CLASSES = {'concat' : BertForPronounResolution_Concat,
                'mul' : BertForPronounResolution_Mul,
                'segment' : BertForPronounResolution_Segment
                }

def initialize(args):
    
    # Create datasets

    train_df =  pd.read_pickle('train_processed.pkl')
    val_df =  pd.read_pickle('val_processed.pkl')

    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)

    #Create model

    model = MODEL_CLASSES[args.model_type].from_pretrained(args.bert_type)
    if type(model) is BertForPronounResolution_Segment:
        model.post_init()
    
    logger.info(f'Model used = {type(model)}')
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr) #change it to AdamW later

    return model, optimizer, train_dataset, val_dataset

def main(args):
    logger.info('Starting single GPU/CPU training')   
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')
    args.device = device

    model, optimizer, train_dataset, val_dataset = initialize(args)
    
    sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,batch_size= args.batch_size, sampler = sampler)

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,batch_size= args.val_batch_size, sampler = val_sampler)

    metrics = train(train_dataloader, val_dataloader, model, optimizer, args)


def distributed_main(args):
    '''
    Similar to main but sets the mode and data loader for distributed programming.
    '''
    global logger 
    logging.root.handlers = []
    logging.basicConfig(level="INFO", 
                    format = 'Ranks {}: %(asctime)s:%(levelname)s: %(message)s'.format(args.local_rank) ,
                    stream = sys.stdout)
    logger = logging.getLogger(__name__)
    logger.info (args)

    dist_util.ddp_setup(args)

    args.device = torch.device(args.local_rank) # <--
    logger.info(f'device = {args.device}')  
    
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

if __name__ == '__main__':
    args = parser.parse_args()
    args.batch_size = args.per_gpu_batch_size//args.gradient_accumulation
    logger.info (args)

    
    if args.is_distributed:
        dist_util.run_distributed(distributed_main, args)
    else:
        main(args)
