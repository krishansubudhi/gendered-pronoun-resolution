import torch, transformers
from transformers import BertModel, BertConfig,BertPreTrainedModel
import math


# Cancat Model architecture
'''
logits = concat(P_hidden, A_hidden, B_hidden)
'''

class BertForPronounResolution_Concat(BertPreTrainedModel):
    def __init__(self, config : BertConfig):
        super(BertForPronounResolution_Concat, self).__init__(config)
        
        self.bert = BertModel(config)
        
        #[P][A][B] classification layer
        self.classification = torch.nn.Linear(config.hidden_size * 3 , 3)
        
        self.init_weights()
    
    def forward(self, input_ids, attention_mask, pab, labels = None, token_type_ids = None ):
   
        #print(f'input shape = {input_ids.size()}')
        output = self.bert(input_ids, attention_mask, token_type_ids, None, None)
        last_hidden_states = output[0]
        
        batches = last_hidden_states.size()[0]
        row_indexes = torch.arange(batches).unsqueeze(1) # row numbers in a column matrix
        pab_hidden_states = last_hidden_states[row_indexes, pab] #batch size x 3 x hidden size
        
        concatenated_states = pab_hidden_states.view(batches,-1)
        
        #print(concatenated_states)
        logits = self.classification(concatenated_states)
        
        output = (logits,) + output[2:] #hidden states and attention if present
        
        if labels is not None:
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(logits, labels)
            
            output = (loss,) + output
            
        return output


# Mul Model Architecture

'''
A_att, B_att = Attention of P wrt A and B/sqrt(hiddensize)
Neither_att = Attention of P wrt random tranable tensor/sqrt(hiddensize)

logits = A_att, B_att, Neither_att
'''
class BertForPronounResolution_Mul(BertPreTrainedModel):
    def __init__(self, config : BertConfig):
        super(BertForPronounResolution_Mul, self).__init__(config)
        
        self.bert = BertModel(config)  
        self.init_weights()
        
        self.neither_params = torch.nn.Parameter(data = torch.randn(config.hidden_size,))
        #TODO: use a trainable tensor instead of CLS
        #self.trainable_tensor = torch.nn.
    
    def forward(self, input_ids, attention_mask, pab, labels = None, token_type_ids = None ):
   
        #pabc = torch.cat(( pab, torch.zeros_like(pab[:,0:1])), dim =1)[:,[0,3,1,2]]
        
        #print(f'input shape = {input_ids.size()}')
        output = self.bert(input_ids, attention_mask, token_type_ids, None, None)
        last_hidden_states = output[0]
        
        batches = last_hidden_states.size()[0]
        row_indexes = torch.arange(batches).unsqueeze(1) # row numbers in a column matrix
        
        pab_hidden_states = last_hidden_states[row_indexes, pab] #batch size x 3 x hidden size
        #print(pab_hidden_states)
        
        p_state = pab_hidden_states[:,0:1]
        attentions = pab_hidden_states[:,1:] * p_state#batch size x 2 x hidden size
        
        neither_att = p_state * self.neither_params
        
        attentions = torch.cat((neither_att, attentions), dim = 1)
        #print(attentions.size())
        #dividing sum with hidden size length to avoid high values. Done in Bert paper too
        logits = torch.sum(attentions, dim = 2)/math.sqrt(self.config.hidden_size) #batch size x 3 

        #print(logits)
        
        output = (logits,) + output[2:] #hidden states and attention if present
        
        if labels is not None:
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(logits, labels)
            
            output = (loss,) + output
            
        return output

# m = BertForPronounResolution_Mul.from_pretrained('bert-base-uncased')
# a = torch.randint(high = 100,size = (2,5))
# pab = torch.randint(low = 1, high = 5, size = (2,3))
# m(a,None,pab)


# ## Segment ID model architecture

'''
input = same as bert
input embedding ids = 1 for P, 2 For A, 3 for B, 0 for rest. A,B and P are only first tokens of their respective words
logits = CLS_hidden
'''

class BertForPronounResolution_Segment(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPronounResolution_Segment, self).__init__(config)
        self.bert = BertModel(config)
        
        self.classification = torch.nn.Linear(config.hidden_size , 3)
        self.config.type_vocab_size = 4

        self.token_type_embeddings = torch.nn.Embedding(self.config.type_vocab_size, config.hidden_size)
        
        self.init_weights()
        

    def post_init(self):
        self.token_type_embeddings.weight.data[0] = self.bert.embeddings.token_type_embeddings.weight.data[0] #initializing the first onewith trained embeddings.
        self.bert.embeddings.token_type_embeddings = self.token_type_embeddings

    def get_token_type_ids(self,inp,p):
        tokentypes = torch.zeros_like(inp)
        tokentypes[p[0]]=1
        tokentypes[p[1]]=2
        tokentypes[p[2]]=3    
        return tokentypes
        
    def forward(self, input_ids, attention_mask, pab, labels = None, token_type_ids = None ):
   
        #print([ self.get_token_type_ids(inp,p)  for inp,p in zip(input_ids, pab) ])
        token_type_ids = torch.cat([ self.get_token_type_ids(inp,p).unsqueeze(0)  for inp,p in zip(input_ids, pab) ], dim =0)
        
        #print(f'token types = {token_type_ids}')
        
        output = self.bert(input_ids, attention_mask, token_type_ids, None, None)
        
        pooler_output = output[1]

        logits = self.classification(pooler_output)
        #print(logits)
        
        output = (logits,) + output[2:] #hidden states and attention if present
        
        if labels is not None:
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(logits, labels)
            
            output = (loss,) + output
            
        return output

# m = BertForPronounResolution_Segment.from_pretrained('bert-base-uncased')
# print(m.bert.embeddings.token_type_embeddings.weight)
# m.post_init()
# print(m.bert.embeddings.token_type_embeddings.weight)
        
# a = torch.randint(high = 100,size = (2,5))
# pab = torch.IntTensor([[3,1,4],[3,1,2]])
# print(a,pab)
# print(m(a,None,pab))