import pandas as pd
import numpy as np

import torch, os
from torch.utils.data import (Dataset,DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import BertTokenizer
from tqdm import trange

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_tokens_array(row):
    '''
    Splits the text into 4 parts. 
    Splitting is done at P, A and B positions
    Converts the split texts in to tokens.
    A,P,B can span over multiple tokens too. Their positions are not fixed either

    Returns
    array of tokens  [[t1,t2][A1 A2 t3 t4] [P1 P2 t5] [B1 B2 B3 t6 t7] ]
    order of P,A and B positions. (a,p,b)
    '''
    p_offset = row['Pronoun-offset']
    a_offset = row['A-offset']
    b_offset = row['B-offset']
    offsets = {'p':p_offset,'a':a_offset,'b':b_offset}
    
    keys = sorted(offsets,key=lambda z : offsets[z])
    positions = sorted(offsets.values())
    lengths = [positions[i] - positions[i-1] if i >0 else pos for i,pos in enumerate(positions)]
    
    text = row['Text']
    positions.append(len(text))
    
    all_tokens = []
    prev_pos = 0
    for pos in positions:
        subtext = text[prev_pos:pos]
        prev_pos = pos
        tokens = tokenizer.tokenize(subtext)
        all_tokens.append(tokens)
    
    return all_tokens, keys



def get_token_pos(all_tokens, keys):
    '''
    Extracts token positions in morged list
    '''
    token_positions = {}
    prev = 0
    for i,tokens in enumerate(all_tokens[:-1]):
        position = prev+len(tokens)
        prev = position
        token_positions[keys[i]] = position
    return token_positions


def get_tokens_with_positions(row):
    '''
    Returns 
    final_tokens : Text onverted to tokens
    token_positions : dictionary containing token position for a,b and p
    
    Example:
    index = 5
    print(train_df.iloc[index])
    final_tokens, token_positions = get_tokens_with_positions(train_df.iloc[index])
    print(final_tokens, token_positions)
    [final_tokens[v] for v in token_positions.values()]
    '''
    tokens_arr, keys = get_tokens_array(row)
    
    final_tokens = []
    for tokens in tokens_arr:
        final_tokens = final_tokens + tokens 
    
    token_positions = get_token_pos(tokens_arr, keys)
    return final_tokens, token_positions




def create_features(df, haslabels = True):
    '''
    Output Format:
    
    text_tokens	Pindex,Aindex,Bindex	label(0,1,2)

    labels:
    0= Neither
    1 = A
    2 = B
    '''
    processed_df = pd.DataFrame()
    
    for i in trange(len(df)):
        #print(i)
        row = df.iloc[i]
        final_tokens, token_positions = get_tokens_with_positions(row)
        
        assert(final_tokens[token_positions['p']] in row['Pronoun'].lower()), i
        #Assert fails for unicode characters which get converted to ascii --- Raúl García
        #assert(final_tokens[token_positions['a']] in row['A'].lower()), i
        #assert(final_tokens[token_positions['b']] in row['B'].lower()), i 
        
        pab_position = [token_positions[key] for key in 'pab']
        
        if haslabels:
            label = 1 if row['A-coref'] else ( 2 if row['B-coref'] else 0)
        else:
            label = 0
        
        processed_df = processed_df.append({'input':np.array(final_tokens), 'pab_pos':np.array(pab_position), 'label':int(label)}, ignore_index=True)
    return processed_df

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=".", type = str,
                        help = 'Input data dir' )

    parser.add_argument('--output_dir', default=".", type = str,
                        help = 'Output data dir' )

    return parser.parse_args()

if __name__ == '__main__':
    print('hello')
    print(os.listdir())

    args = parse_args()
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)

    train_file = os.path.join(args.input_dir,'gap-coreference/gap-development.tsv')
    val_file = os.path.join(args.input_dir,'gap-coreference/gap-validation.tsv')
    test_file = os.path.join(args.input_dir,'gap-coreference/gap-test.tsv')

    train_df = pd.read_csv(train_file,sep = '\t')
    val_df = pd.read_csv(val_file,sep = '\t')

    print('Processing validation data')
    processed_df = create_features(val_df)
    #for reading purpose
    #saving to TSV will not store the data types of ndarray. It converts them to str
    processed_df.to_csv(os.path.join(args.output_dir,'val_processed.tsv'), sep='\t')
    processed_df.to_pickle(os.path.join(args.output_dir,'val_processed.pkl'))
    df1 = pd.read_pickle(os.path.join(args.output_dir,'val_processed.pkl'))

    print('Processing training data')
    processed_df = create_features(train_df)
    processed_df.to_csv(os.path.join(args.output_dir,'train_processed.tsv'), sep='\t')
    processed_df.to_pickle(os.path.join(args.output_dir,'train_processed.pkl'))
    df2 = pd.read_pickle(os.path.join(args.output_dir,'train_processed.pkl'))

    #concatenated dataframe for final training
    combined_df = pd.concat([df1, df2], ignore_index= "true")
    combined_df.to_pickle(os.path.join(args.output_dir,'train_val_combined.pkl'))

    print(os.listdir(args.output_dir))
    print(f'combined_df length = {len(combined_df)}')

    

