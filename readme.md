This is the solution to the famous Kaggle comptetion.
https://www.kaggle.com/c/gendered-pronoun-resolution/overview

The pytorch based solution utilizes BERT architecture to create a finetuned model.

Training and validation data: https://github.com/google-research-datasets/gap-coreference.git

# Goals

1. ~Create a working solution with good accuracy.~
2. ~Compare different approaches.~
3. Apply Distributed training using pytorch Distributed Data Parallel and Horovod. Document speed improvement.
4. Use NVIDIA apex library for 16 bit floating point precission (fp16). Show training speed and metrics.
5. Use azure ML to train. 
6. Use hyperdrive.

# Steps

1. Boot a machine with at least one GPU

1. Clone the repo
```
git clone https://github.com/krishansubudhi/gendered-pronoun-resolution.git
```
2. `cd gendered-pronoun-resolution`
3. Download data

```
git clone https://github.com/google-research-datasets/gap-coreference.git
``` 

4. Preprocess: Convert text to tokens. Also find P , A and B position. Also add labels. The preprocessed data is stored as both tsv and pkl files. pkl file will be loaded for traning while tsv file is for readablility.

```
python PreprocessGapData.py
```

5. Run code in single GPU with default configurations.
```
python TrainGAP.py
```


# Solution

I have tried 3 types of architectures to solve this problem using Bert. To find the best among them Bert Base is used as the bert model.
Results can be obtained by running the **Experimentation** notebook.

## 1. Concatenation

logits = linear(concat(P_hidden, A_hidden, B_hidden))

**Results**
Epoch = 0, Val loss = 0.4355, val_acc = 0.8515 

## 2. Multiplication

This attends P over A,B start tokens and a random tensor for neigher case. Softmax is calculated over the logits

A_att, B_att = Attention of P wrt A and B/sqrt(hiddensize)
Neither_att = Attention of P wrt random tranable tensor/sqrt(hiddensize)

logits = A_att, B_att, Neither_att

**Results**
Epoch = 1, Val loss = 0.16755544440820813, val_acc = 0.9535 

This method produces the best results so far. And this will be our baseline. 
Further hyper parameter tuning will be done on top of this architecture.

## 3. Segmentation

input = same as bert
input embedding ids = 1 for P, 2 For A, 3 for B, 0 for rest. 0 token type embeddings are initialized with pretrained weights. Rest of the weights are random.

A,B and P are only first tokens of their respective words
logits = linear(CLS_hidden)

**Results**
Epoch = 1, Val loss = 0.8980, val_acc = 0.596   

This performs the worst among the three. Probable casue can be the addition of many untrained weights at the beginning of the pretrained model. Or segment ids are not useful at all. It needs further debugging.


#  Distributed training

There are multiple ways to do distributed traing. For example pytorch DP, DDP and horovod are different libraries. Then there is nccl and gloo backends. Also there are multiple ways to start multi process training. We will discuss all of them and cgive a comparison of all.

For all the setting, these are the constants.

Total batch size for 1 optimizer step = 32
epochs = 1
learning ratee = 2E-5

**Single node performance**

Training loss |Time taken for epoch 1 |Val loss, val_acc |
 --- | --- | --- |
 0.28 | 290 s|0.1776, 0.9495 |


## 1. Manually starting multiple process
Refer the notebooks Multiprocess1 and Multiprocess2 for running disstributed data parallel training using 2 processes. Needs a VM with 2 GPUS.

### Performance with single node 2 processes
|Backend|Training loss |Time taken for epoch 1 |Val loss, val_acc |
|--- | --- | --- | --- |
|GLOO| 0.441, 0.301 | 218 s|0.158, 0.9495 |
|NCCL| 0.549, 0.212 | 179 s |0.183, 0.945 |

Conclusion: NCCL is faster then GLOO. NCCL with two processes only took 61% of single node time while GLOO took 75% of single node time which is still faster but not very efficient.

### Performance with multi node 2 processes
