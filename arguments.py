import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', default=1, type = int, required=False,
                    help = 'Number of epochs' )
parser.add_argument('--lr', default=2E-5, type = float, required= False,
                    help = 'Learning Rate')

parser.add_argument('--per_gpu_batch_size',default= 32, type = int )

parser.add_argument('--val_batch_size',default= 64, type = int )

parser.add_argument('--model_type', default = 'mul', type = str,
                    help = 'Model type from concat, mul or segment')
parser.add_argument('--bert_type', default = 'bert-base-uncased', type = str,
                    help = 'Bert model type')
parser.add_argument('--gradient_accumulation', default = 8, type = int)

#DDP arguments                    
parser.add_argument('--is_distributed',  action='store_true')
parser.add_argument('--world_size', default = None, type = int, help = 'Total number of processes') #needed for spawn to
parser.add_argument('--backend', default = 'gloo', type = str)
parser.add_argument('--master_node', default = 'localhost', type = str)
parser.add_argument('--master_port', default = 12533, type = int)

#SPAWN arguemtns
parser.add_argument('--nprocs', default = 1, type = int, help = 'Number of processes to spawn')
parser.add_argument('--start_rank', default = 0, type = int, help = 'starting rank for this node')

#Explicit
parser.add_argument('--local_rank', default = -1, type = int) #also passed in torch.distrib.launch
parser.add_argument('--global_rank', default = None, type = int)

#horovod
parser.add_argument('--use_horovod', action="store_true", default=False)

#fp16
# https://nvidia.github.io/apex/amp.html
parser.add_argument('--fp16', action="store_true", default=False)
parser.add_argument('--amp_opt_level', default="00", type = str)