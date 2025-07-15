import argparse
import os
import random
import wandb
import numpy as np
import torch
import torch.backends.cudnn as cudnn

def change_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True