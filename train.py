
import argparse
import os
import random
import wandb
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import ivcr.tasks as tasks
from ivcr.common.config import Config
from ivcr.common.dist_utils import get_rank, init_distributed_mode, is_main_process
from ivcr.common.logger import setup_logger
from transformers import AutoTokenizer
from ivcr.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
# from utils.logger import setup_logger
from ivcr.common.registry import registry
from ivcr.common.utils import now

# imports modules for registration
import sys
from ivcr.datasets.builders import *
from ivcr.models import *
from ivcr.processors import *
from ivcr.runners import *
from ivcr.tasks import *
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
os.environ["ACCELERATE_USE_DDP"] = "true"
os.environ["ACCELERATE_DDP_FIND_UNUSED_PARAMETERS"] = "true"
def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args
def setup_seeds(seed: int, accelerator=None):
    if accelerator is not None:
        seed += accelerator.process_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# def setup_seeds(config):
#     seed = config.run_cfg.seed  + get_rank()
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     cudnn.benchmark = False
#     cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    setup_logger()
    job_id = now()
    cfg = Config(parse_args())
    accelerator = Accelerator(gradient_accumulation_steps=cfg.run_cfg.accum_grad_iters)
 
    accelerator.wait_for_everyone()
    setup_seeds(cfg.run_cfg.seed, accelerator=accelerator)
    # setup_seeds(cfg)
    # cfg.pretty_print()

    task = tasks.setup_task(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.config.model.llama_model)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = task.build_model(cfg,tokenizer)
    datasets = task.build_datasets(cfg,tokenizer)
    

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets,accelerator=accelerator
    )
    runner.train()


if __name__ == "__main__":
    main()
