
# 2 nodes 2 processes

### Node 1
Not working Yet


```python
!python TrainGAP.py \
--is_distributed \
--local_rank 0 \
--global_rank 0 \
--world_size 2 \
--master_node '40.74.233.84' \
--master_port 29500 \
--backend nccl \
--per_gpu_batch_size 16 \
--gradient_accumulation 4
```

### Node 2


```python
# !python TrainGAP.py \
# --is_distributed \
# --local_rank 0 \
# --global_rank 1 \
# --world_size 2 \
# --master_node '40.74.233.84' \
# --master_port 29500 \
# --backend nccl \
# --per_gpu_batch_size 16 \
# --gradient_accumulation 4
```

**Common errors**

1. Training does not progress or timeout error is seen

    Follow this blog to test if distributed training is possible between the machines or not

    https://krishansubudhi.github.io/deeplearning/2019/10/15/PyTorch-Distributed.html

2. Address already in use

    Same port was used for multi node communication which has not been released. 

    Restart kernel or kill any other distributed process which uses same port . 

# 2 nodes 4 processes
## using mp.spawn

### Node 1


```python
!python TrainGAP.py \
        --is_distributed \
        --world_size 4 \
        --nprocs 2 \
        --start_rank 0\
        --backend nccl \
        --master_node '40.74.233.84' \
        --master_port 29500 \
        --per_gpu_batch_size 8 \
        --gradient_accumulation 2
```

### Node 2


```python
# !python TrainGAP.py \
#         --is_distributed \
#         --world_size 4 \
#         --nprocs 2 \
#         --start_rank 2\
#         --backend nccl \
#         --master_node '40.74.233.84' \
#         --master_port 29500 \
#         --per_gpu_batch_size 8 \
#         --gradient_accumulation 2
```

## Using torch.distributed.launch

### Node 1


```python
!python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr='40.74.233.84' \
    --master_port=29500 \
    TrainGAP.py \
        --is_distributed \
        --per_gpu_batch_size 16 \
        --gradient_accumulation 4 \
        --backend nccl
```

### Node 2


```python
# !python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     --nnodes=2 \
#     --node_rank=1 \
#     --master_addr='40.74.233.84' \
#     --master_port=29500 \
#     TrainGAP.py \
#         --is_distributed \
#         --per_gpu_batch_size 16 \
#         --gradient_accumulation 4 \
#         --backend nccl
```
