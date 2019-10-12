#!pip install transformers

import pandas as pd
import numpy as np
import transformers
from transformers import BertPreTrainedModel, BertTokenizer, BertModel, BertConfig
import torch
from torch.utils.data import DataLoader,TensorDataset, RandomSampler, SequentialSampler
import math
from tqdm import trange
from tqdm import tqdm as tqdm
from BertModels import *
from arguments import parser

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

    #print(ids.size(), masks.size(), pabs.size(), labels.size())

    return TensorDataset(ids, masks, pabs, labels)



def evaluate(val_dataloader,model):
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

    for epoch in tqdm(range(args.epochs),position=1, total=args.epochs):
        print(f'Training epoch {epoch}')
        model.train()
        batch_iterator = tqdm(train_dataloader, desc='batch_iterator')
        
        for step, batch in enumerate(batch_iterator):
            batch = (t.to(args.device) for t in batch)
            loss,logits = model(*batch)

            #print(f'step = {step}, loss = {losses[-1]}')

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
        print(f'Evaluating for epoch {epoch+1}')
        val_loss , val_acc = evaluate(val_dataloader, model)
        batch_iterator.write(f'Epoch = {epoch+1}, Val loss = {val_loss}, val_acc = {val_acc}')
        
    return losses, val_loss , val_acc

    
MODEL_CLASSES = {'concat' : BertForPronounResolution_Concat,
                'mul' : BertForPronounResolution_Mul,
                'segment' : BertForPronounResolution_Segment
                }

def main(args):
    
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Create datasets

    train_df =  pd.read_pickle('train_processed.pkl')
    val_df =  pd.read_pickle('val_processed.pkl')

    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)

    #Create model

    model = MODEL_CLASSES[args.model_type].from_pretrained(args.bert_type)
    if type(model) is BertForPronounResolution_Segment:
        model.post_init()
    print(f'Model used = {type(model)}')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr) #change it to AdamW later
    sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,batch_size= args.batch_size, sampler = sampler)

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,batch_size= args.val_batch_size, sampler = val_sampler)

    metrics = train(train_dataloader, val_dataloader, model, optimizer, args)

if __name__ == '__main__':
    args = parser.parse_args()
    args.batch_size = args.per_gpu_batch_size//args.gradient_accumulation
    print (args)

    print('Starting single GPU/CPU training')
    
    #if args.is_distributed:
        # For distributed training using DDP, there are 3 types of init methods.
        # 1. pass ranks, master node address and world size explicitly
        # 2. Do mp.spawn with start rank and end rank.
        # 3. torch.distributed.launch which automatically sets some variables.
        # https://github.com/huggingface/transformers/blob/a701c9b32126f1e6974d9fcb3a5c3700527d8559/transformers/modeling_bert.py#L177
        # https://github.com/pytorch/fairseq/blob/d80ad54f75186adf9b597ef0bcef005c98381b9e/fairseq/distributed_utils.py#L71
    #else
    main(args)
