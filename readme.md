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
Epoch = 1, Val loss = 0.3918, val_acc = 0.8546

This method produces the best results so far. And this will be our baseline. 
Further hyper parameter tuning will be done on top of this architecture.

## 3. Segmentation

input = same as bert
input embedding ids = 1 for P, 2 For A, 3 for B, 0 for rest. 0 token type embeddings are initialized with pretrained weights. Rest of the weights are random.

A,B and P are only first tokens of their respective words
logits = linear(CLS_hidden)

**Results**
Epoch = 1, Val loss = 1.003, val_acc = 0.5176

This performs the worst among the three. Probable casue can be the addition of many untrained weights at the beginning of the pretrained model. Or segment ids are not useful at all. It needs further debugging.


#  Distributed training

There are multiple ways to do distributed traing. For example pytorch DP, DDP and horovod are different libraries. Then there is nccl and gloo backends. Also there are multiple ways to start multi process training. We will discuss all of them and cgive a comparison of all.

For all the setting, these are the constants.

Total batch size for 1 optimizer step = 32
epochs = 1
learning ratee = 2E-5
Machine used = Azure NC12_promo
torch.cuda.get_device_name(0)
'Tesla K80'

Performance is a bit slow in the K80 GPUs. Will benchmark in V100s after AzureML integration.

## Single node single process performance


GPU | Training loss |Time taken for epoch 1 |Val loss, val_acc |
--- | --- | --- | --- |
K80 | 0.243 | 389 s|0.3918, 0.8546 |
V100 | 0.28 | 290 s| - |

## Starting multiple process
Refer the notebooks Multiprocess1 and Multiprocess2 for running disstributed data parallel training using 2 processes. Needs a VM with 2 GPUS.

Also the notebook mp_spawn shows easier way to start multiple processes in one command using pytorch multiprocess spawn method.

### Performance with single node 2 processes

K80 GPUS

|Backend|Training loss (node1,node2) |Time taken for epoch 1 |Val loss, val_acc |
|--- | --- | --- | --- |
|GLOO| 0.409, | 256 s|0.449, 0.8458 |
|NCCL| 0.409, | 212 s |0.449, 0.8458 |

V100 GPUS

|Backend|Training loss (node1,node2) |Time taken for epoch 1 |Val loss, val_acc |
|--- | --- | --- | --- |
|GLOO| 0.441, 0.301 | 218 s| - |
|NCCL| 0.549, 0.212 | 179 s | - |

Conclusion: Multiprocess traning is faster than single process training. NCCL is faster then GLOO. In V100s, NCCL with two processes only consumed 61% of single node time while GLOO took 75% of single node time which is still faster but not very efficient.

Hence rest of the experiments will be done with NCCL backend only.

### Performance with multi node 

For multi node training I booted two azure VMs in same location and resource group. I opened port 29500 in the VM I was going to use for rank 0. More details on this [blog](https://krishansubudhi.github.io/deeplearning/2019/10/15/PyTorch-Distributed.html)

K80

|Nodes,Processes|Training loss |Time taken for epoch 1 |Val loss, val_acc |
|--- | --- | --- | --- |
|2 nodes, 2 processes| 0.43, 0.267 | 333 s |0.416, 0.841 |
|2 nodes, 4 processes| 0.434, 0.16, 0.091, 0.764 | 204 s | 0.419, 0.8458 |

2 nodes 2 process decreased the speed while 2 nodes 4 process improved the speed. The intuition is with more number of processes, the communicaiton time is constant but the parallelisation improved. But this needs more analysis.

Previously with 4 processes my the performance has degraded. The reson was I was not reducing batch size per GPU hence the overall batch size was consant. Since I kept my learning rate constant, the higher batch performance was bad. The current result has per gpu batch size = 8 for 4 processes. and 16 for 2 processes making the overall batch size = 32 in both the cases.

**Instructions to run multinode tranining:**

https://github.com/krishansubudhi/gendered-pronoun-resolution/blob/master/MultiNode.md
