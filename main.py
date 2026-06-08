import os
import torch
import math
from exp.exp_main_PINN import exp_main
import numpy as np
import random

def seed_everything(seed=11, use_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

train_battery = '4'
start = 20
name = 'B20'
task = 'SOH'
battery = 'B20'
pred_len = 1
window_size = 10

class Args:
    def __init__(self):
        self.model = 'PINN'
        self.model_id = '{}_{}_{}#{}'.format(window_size,self.model,name,train_battery)
        self.task = task
        self.battery = battery
        self.results_path = './results/{}/'.format(battery)
        self.checkpoints = './checkpoints/'
        self.root_path = './datasets/{}/'.format(battery)
        self.train_battery = train_battery
        self.train_battery_now = train_battery
        self.data_path = 'battery_data_frames[].csv'
        self.start = start
        self.norm = True
        self.battery_EOL = battery_EOL

        self.epochs = 60  #60
        self.patience = 40
        self.optim = 'adam'

        self.warmup_epochs = 10
        self.min_lr = 0
        self.smoothing_learning_rate = 0
        self.damping_learning_rate = 0
        self.lradj = 'exponential_with_warmup'
        self.alpha = 0.1
        self.beta = 1
        self.rated_capacity = rated_capacity

        self.learning_rate = 0.0005  # 0.0001
        self.batch_size = 32  #16
        self.d_ff = 128  #4
        self.d_mlp = 2048  #32
        self.d_model = 512   #1024
        self.n_heads = 4  #1
        self.e_layers = 2  #6
        self.d_layers = 1
        self.dropout = 0.1
        self.d_state = 256
        self.window_size = window_size

        self.activation = 'relu'
        self.output_attention = False
        self.pred_len = pred_len
        self.seq_len = 12
        self.label_len = self.seq_len
        self.c_out = 1

        self.grad_clip = 1.0
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'

args = Args()

if not args.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

seed_everything(11, use_gpu=args.use_gpu)
exp = exp_main(args)
print('<<<<<<<<<<<<<<<<<<<<<<<<< start training >>>>>>>>>>>>>>>>>>>>>>>>>')
exp.train()
# exp.test(Time_record=False)

if args.use_gpu and torch.cuda.is_available():
    torch.cuda.empty_cache()
