import torch
import transformers
from torch.utils.data import TensorDataset

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

def create_dataset(df,cache_dir, bert_type):
    tokenizer = transformers.BertTokenizer.from_pretrained( bert_type,cache_dir = cache_dir )
    features = [get_features_from_example(df.iloc[i],tokenizer) for i in range(len(df))]

    ids = torch.tensor([feature[0] for feature in features])
    masks = torch.tensor([feature[1] for feature in features])
    pabs = torch.tensor([feature[2] for feature in features], dtype = torch.long)
    labels = torch.tensor([feature[3] for feature in features])

    #logger.info(ids.size(), masks.size(), pabs.size(), labels.size())

    return TensorDataset(ids, masks, pabs, labels)