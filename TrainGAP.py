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

def get_features_from_example(ex):
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
    features = [get_features_from_example(df.iloc[i]) for i in range(len(df))]

    ids = torch.tensor([feature[0] for feature in features])
    masks = torch.tensor([feature[1] for feature in features])
    pabs = torch.tensor([feature[2] for feature in features])
    labels = torch.tensor([feature[3] for feature in features])

    print(ids.size(), masks.size(), pabs.size(), labels.size())

    return TensorDataset(ids, masks, pabs, labels)



def evaluate(val_dataset,model):
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,batch_size= args.val_batch_size, sampler = val_sampler)

    all_labels = []
    all_preds = []
    total_loss = 0
    acc = 0
    model.eval()
    steps = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            labels = batch[-1]
            batch = tuple(t.to(device) for t in batch)
            loss,logits = model(*batch)
            preds = torch.argmax(logits, dim = 1)

            total_loss+=loss.item()
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            steps += 1


    acc = np.sum(np.array(all_preds) == np.array(all_labels))/ len(all_preds)
    loss = total_loss/steps
    return loss,acc

def train(train_dataset, val_dataset, model, args):
    
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr) #change it to AdamW later
    sampler = RandomSampler(train_dataset)
    dataloader = DataLoader(train_dataset,batch_size= args.batch_size, sampler = sampler)

    losses = []
    total_loss = 0

    for epoch in tqdm(range(args.epochs),position=1, total=args.epochs):
        print(f'Training epoch {epoch}')
        model.train()
        batch_iterator = tqdm(dataloader, desc='batch_iterator')
        
        for step, batch in enumerate(batch_iterator):
            batch = (t.to(device) for t in batch)
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
        print(f'Evaluating for epoch {epoch}')
        val_loss , val_acc = evaluate(val_dataset, model)
        batch_iterator.write(f'Epoch = {epoch}, Val loss = {val_loss}, val_acc = {val_acc}')
        
    return losses

    
MODEL_CLASSES = {'concat' : BertForPronounResolution_Concat,
                'mul' : BertForPronounResolution_Mul,
                'segment' : BertForPronounResolution_Segment
                }

args = parser.parse_args()
args.batch_size = args.per_gpu_batch_size//args.gradient_accumulation


print (args)

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device(0)

cls_id = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
sep_id = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
max_length = 512

# Create datasets

train_df =  pd.read_pickle('train_processed.pkl')
val_df =  pd.read_pickle('val_processed.pkl')

train_dataset = create_dataset(train_df)
val_dataset = create_dataset(val_df)

#Create model

model = MODEL_CLASSES[args.model_type].from_pretrained(args.bert_type)
print(f'Model used = {type(model)}')
model = model.to(device)
losses = train(train_dataset, val_dataset, model, args)