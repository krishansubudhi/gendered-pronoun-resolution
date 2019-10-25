# For distributed training using DDP, there are 3 types of init methods.
# 1. pass ranks, master node address and world size explicitly
# 2. Do mp.spawn with start rank and end rank.
# 3. torch.distributed.launch which automatically sets env variables.
# https://github.com/huggingface/transformers/blob/a701c9b32126f1e6974d9fcb3a5c3700527d8559/transformers/modeling_bert.py#L177
# https://github.com/pytorch/fairseq/blob/d80ad54f75186adf9b597ef0bcef005c98381b9e/fairseq/distributed_utils.py#L71
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sys,logging
import os,time
from TrainGAP import distributed_main, distributed_main_horovod

logger = logging.getLogger(__name__)

def run_distributed(args):
    '''
    Check if horovod is used to start the process.
    Check if distributed variables are set properly. If not start a multiprocess module. 
    Else set the variables in args and call distributed_main(). 
    local_rank, global_rank, world_size, master_node, master_port
    '''
    if args.use_horovod:
        distributed_main_horovod(args)
    elif args.local_rank > -1:
        if args.global_rank is not None:
            #explicit ranks set
            distributed_main(args)
        else:
            #torch.distrib.launch
            update_args_from_env(args)
            distributed_main(args)
    else:
        #spawn processes
        #https://pytorch.org/docs/stable/distributed.html#launch-utility
        #https://github.com/pytorch/examples/blob/master/imagenet/main.py
        mp.spawn(spawn_fn, nprocs=args.nprocs, args = (distributed_main, args)) #spawn also sends the local_rank as first argument

def update_args_from_env(args):
    #https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py
    logger.info(f'Updating args from environment variable. Rank = {args.local_rank}')
    current_env = os.environ
    
    print ('Current environment = ',current_env)
    
    args.global_rank = int(current_env["RANK"])
    args.master_node = current_env["MASTER_ADDR"]
    args.master_port = int(current_env["MASTER_PORT"])
    args.world_size = int(current_env["WORLD_SIZE"])

def spawn_fn(local_rank, distributed_main, args):
    logger.info(f'inside spawn method . Rank = {local_rank}')
    args.local_rank = local_rank
    args.global_rank = args.start_rank + local_rank
    distributed_main(args)

def ddp_setup(args):
    #required = ['backend', 'local_rank', 'global_rank', 'world_size', 'master_node','master_port']
    logger.info('Setting up DDP') 
    dist.init_process_group(backend  = args.backend,
                        rank = args.global_rank,
                        world_size = args.world_size,
                        init_method=f'tcp://{args.master_node}:{args.master_port}') 
                        #see if it can be set to something like env:// for local machines
    torch.manual_seed(42) #otherwise model initialization will not be uniform. Not tested 
