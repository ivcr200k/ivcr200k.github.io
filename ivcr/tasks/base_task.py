"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch.nn as nn
import copy
import os
import wandb
from accelerate import Accelerator
import time
import torch
import torch.distributed as dist
from ivcr.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from ivcr.common.logger import MetricLogger, SmoothedValue
from ivcr.common.registry import registry
from ivcr.common.constant import change_seeds
from utils.logger import setup_logger
from ivcr.datasets.data_utils import prepare_sample


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg,tokenizer):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config,tokenizer)

    def build_datasets(self, cfg,tokenizer):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."
        for name in datasets_config:
            if name == "charades_instruct":
                continue
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config,tokenizer)
            dataset = builder.build_datasets()
            
            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset
        
        return datasets

    def train_step(self, model, samples,flag):
        loss = model(samples,flag=flag)["loss"]
        return loss

    def eval_step(self,model,samples, flag):
        output = model(samples,flag=flag)
        return output

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        accelerator,
        data_loader,
        optimizer,
        lr_scheduler,
        flag,
        scaler=None,
        cuda_enabled=False,
        log_freq=100,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
            flag = flag,
            accelerator= accelerator
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        flag,
        model,
        accelerator,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=1,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        
        data_loader, optimizer, model = accelerator.prepare(data_loader,optimizer,model)
        use_amp = scaler is not None
        
        logger = logging.getLogger('ivcr')
        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logger.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)


        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header,logger):
            if i >= iters_per_epoch:
                break
            samples = next(data_loader)
            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)
            
            with accelerator.accumulate(model):
                loss = self.train_step(model=model, samples=samples,flag=flag)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            # logger.info({'train/loss': loss.item(),
            #                'train/lr': optimizer.param_groups[0]["lr"]})
            # if i%500 == 0 and i!=0:
            #     model.eval()
            #     output = self.eval_step(model=model, samples = samples,flag = 3)
            #     print("验证模型的输出:")
            #     print(output)
            #     model.train()
            # samples.update(
            #     {
            #         "epoch": inner_epoch,
            #         "num_iters_per_epoch": iters_per_epoch,
            #         "iters": i,
            #     }
            # )
            # with torch.cuda.amp.autocast(enabled=use_amp):
            #     loss = self.train_step(model=model, samples=samples,flag=flag)
            # if use_amp:
            #     scaler.scale(loss).backward()
            # else:
            #     loss.backward()
            # update gradients every accum_grad_iters iterations
            # if (i + 1) % accum_grad_iters == 0:
            #     if use_amp:
            #         scaler.step(optimizer)
            #         scaler.update()                     
            #     else:
            #         optimizer.step()
            #     optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if is_main_process() and wandb.run is not None:
                wandb.log({'train/loss': loss.item(),
                           'train/lr': optimizer.param_groups[0]["lr"]},
                          step=epoch * iters_per_epoch + i)

        metric_logger.synchronize_between_processes()
        logger.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
