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
parser.add_argument('--local_rank', default = -1, type = int)
parser.add_argument('--global_rank', default = None, type = int)
parser.add_argument('--world_size', default = None, type = int)
parser.add_argument('--backend', default = 'gloo', type = str)
parser.add_argument('--master_node', default = 'localhost', type = str)
parser.add_argument('--master_port', default = 12533, type = int)
