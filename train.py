# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import time
import os
from typing import Tuple

from warnings import warn

import numpy as np
import torch
import wandb
import torch.distributed as dist
from tqdm import tqdm
import os.path as osp
from datetime import timedelta

from utilities.configs import TrainingConfig, ScheduleMethods, LossNames, LogNames, LogTypes
from loss_metrics.loss import Loss
from utilities.functions import to_device, get_device, release_cuda
from utilities.common import BasePipeline
from data.data_loader import train_data_loader, evaluation_data_loader


class Trainer(BasePipeline):
    def __init__(self, cfgs: TrainingConfig):
        """
        This class is the trainner
        Args:
            cfgs: the configuration of the training class
        """
        super().__init__(cfgs)
        self.distribute_evaluation = cfgs.distribute_evaluation
        self.max_epoch = cfgs.max_epoch
        self.evaluation_freq = cfgs.evaluation_freq
        self.no_log = cfgs.no_log
        self.output_dir = cfgs.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.iteration = 0
        self.epoch = 0
        self.training = False

        # set up loggers
        if not self.no_log:
            self._build_logger(cfgs=cfgs)

        self._build_optimizers(cfgs=cfgs)

        # loss functions
        if self.device == "cuda":
            self.loss_func = Loss(config=cfgs.loss, device=self.device, datatype=cfgs.data.data_type,
                                  model_type=cfgs.model.type).cuda()
        else:
            self.loss_func = Loss(config=cfgs.loss, device=self.device, datatype=cfgs.data.data_type,
                                  model_type=cfgs.model.type).to(self.device)

        # datasets:
        self.training_data_loader = train_data_loader(cfg=cfgs.data, model_type=cfgs.model.type)
        self.evaluation_data_loader = evaluation_data_loader(cfg=cfgs.data, model_type=cfgs.model.type,
                                                             distribute_evaluation=self.distribute_evaluation)
        self.best_metric = 0
        # self.save_snapshot("debug/test_model.tar")
        # print("test")

    def _build_logger(self, cfgs):
        configs = {
            "lr": cfgs.lr,
            "lr_t0": cfgs.lr_t0,
            "lr_tm": cfgs.lr_tm,
            "lr_min": cfgs.lr_min,
            "gpus": cfgs.gpus,
            "epochs": self.max_epoch
        }
        wandb.login(key=cfgs.wandb_api)
        if self.distributed:
            self.wandb_run = wandb.init(project=self.name, config=configs, group="DDP", entity=cfgs.wandb_proj)
        else:
            self.wandb_run = wandb.init(project=self.name, config=configs, entity=cfgs.wandb_proj)

    def cleanup(self):
        dist.destroy_process_group()
        self.wandb_run.finish()

    def optimizer_step(self):
        """
        run one step of the optimizer
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def step(self, data_dict, train=True) -> Tuple[dict, dict]:
        """
        one step of the model, loss function and also the loss_metrics
        Args:
            train:
            data_dict: the input data dictionary
        Returns:
            the output from the model, the output from the loss function
        """
        # start_time = time.time()
        # data_dict = to_device(data_dict, device=self.device)
        output_dict = self.model(data_dict)
        torch.cuda.empty_cache()
        if train:
            loss_dict = self.loss_func(output_dict=output_dict, input_dict=data_dict)
        else:
            loss_dict = self.loss_func.evaluate(output_dict=output_dict, input_dict=data_dict)
        torch.cuda.empty_cache()
        return loss_dict

    def update_log(self, results, timestep=None, log_name=None):
        if not self.no_log:
            if timestep is not None:
                self.wandb_run.log({LogNames.step_time: timestep})
            if log_name == LogTypes.train:
                value = self.scheduler.get_last_lr()
                self.wandb_run.log({log_name + "/" + LogNames.lr: value[-1]})

            if log_name is None:
                for key, value in results.items():
                    self.wandb_run.log({key: value})
            else:
                for key, value in results.items():
                    self.wandb_run.log({log_name + "/" + key: value})

    def run_epoch(self):
        """
        run training epochs
        """
        self.optimizer.zero_grad()

        last_time = time.time()
        # with open(self.output_file, "a") as f:
        #     print("Training CUDA {} Epoch {} \n".format(self.current_rank, self.epoch), file=f)
        for iteration, data_dict in enumerate(
                tqdm(self.training_data_loader, desc="Training Epoch {}".format(self.epoch))):
            self.iteration += 1

            output_dict = self.step(data_dict=data_dict)
            torch.cuda.empty_cache()

            output_dict[LossNames.loss].backward()
            self.optimizer_step()
            optimize_time = time.time()

            output_dict = release_cuda(output_dict)
            self.update_log(results=output_dict, timestep=optimize_time-last_time, log_name=LogTypes.train)
            last_time = time.time()
        self.scheduler.step()

        if not self.distributed or (self.distributed and self.current_rank == 0):
            os.makedirs('{}/models'.format(self.output_dir), exist_ok=True)
            self.save_snapshot('{}/models/{}_{}.pth'.format(self.output_dir, self.name, self.epoch))

    def inference_epoch(self):
        self.loss_func.metric.reset()
        output_dict = {}
        if (self.evaluation_freq > 0) and (self.epoch % self.evaluation_freq == 0): # and (self.epoch != 0):
            average_time_stamp = []
            for iteration, data_dict in enumerate(tqdm(self.evaluation_data_loader,
                                                       desc="Evaluation Losses Epoch {}".format(self.epoch))):
                # if iteration % self.max_evaluation_iteration_per_epoch == 0 and iteration != 0:
                #     break
                start_time = time.time()
                output_dict = self.step(data_dict, train=False)
                torch.cuda.synchronize()
                step_time = time.time()
                output_dict = release_cuda(output_dict)
                torch.cuda.empty_cache()
                average_time_stamp.append(step_time - start_time)
                if self.current_rank == 0:
                    if "mIoU" in output_dict.keys():
                        if output_dict["mIoU"] > self.best_metric:
                            self.best_metric = output_dict["mIoU"]
                            self.save_snapshot(
                                '{}/{}_best_mIoU.pth'.format(self.output_dir, self.name))
                    elif "IoU" in output_dict.keys():
                        if output_dict["IoU"] > self.best_metric:
                            self.best_metric = output_dict["IoU"]
                            self.save_snapshot(
                                '{}/{}_best_IoU.pth'.format(self.output_dir, self.name))
            self.update_log(results=output_dict, timestep=sum(average_time_stamp) / float(len(average_time_stamp)),
                            log_name=LogTypes.others)

    def run(self):
        """
        run the training process
        """
        torch.autograd.set_detect_anomaly(True)
        starting_epoch = copy.deepcopy(self.epoch)
        for self.epoch in range(starting_epoch, self.max_epoch, 1):
            self.set_eval_mode()
            self.inference_epoch()

            self.set_train_mode()
            if self.distributed:
                self.training_data_loader.sampler.set_epoch(self.epoch)
                if self.evaluation_freq > 0 and self.distribute_evaluation:
                    self.evaluation_data_loader.sampler.set_epoch(self.epoch)
            self.run_epoch()
        self.cleanup()
