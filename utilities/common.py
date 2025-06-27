import time
import os
from typing import Tuple

from warnings import warn
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import os.path as osp
from datetime import timedelta

from utilities.configs import TrainingConfig, ScheduleMethods, LossNames, LogNames, LogTypes
from loss_metrics.loss import Loss
from models.model import get_model
from utilities.functions import to_device, get_device, release_cuda
from data.data_loader import train_data_loader, evaluation_data_loader


class BasePipeline:
    def __init__(self, cfgs: TrainingConfig):
        self.name = cfgs.name

        # set up gpus
        # if cfgs.gpus.device == "cuda":
        #     self.device = "cuda"
        # else:
        self.device = get_device(device=cfgs.gpus.device)
        if 'WORLD_SIZE' in os.environ and cfgs.gpus.device == "cuda":
            print("world size: ", int(os.environ['WORLD_SIZE']))
            self.distributed = cfgs.data.distributed = int(os.environ['WORLD_SIZE']) >= 1
            self.device = "cuda:{}".format(int(os.environ['LOCAL_RANK']))
            # log_name = self.name + "-" + str(int(os.environ['WORLD_SIZE'])) + "-" + str(
            #     int(os.environ['LOCAL_RANK'])) + "/" + datetime.now().strftime("%m-%d-%Y-%H-%M")
        # if cfgs.data.distributed and cfgs.gpus.device == "cuda":
        #     self.distributed = cfgs.data.distributed
        else:
            print("world size: ", 0)
            self.distributed = cfgs.data.distributed = False
            # log_name = self.name + "-" + datetime.now().strftime("%m-%d-%Y-%H-%M")

        # model
        self.model = get_model(config=cfgs.model, data_type=cfgs.data.data_type, device=self.device)
        self.snapshot = cfgs.snapshot
        if self.snapshot:
            self.state_dict = self.load_snapshot(self.snapshot)

        self.current_rank = 0
        if self.device == torch.device("cpu"):
            pass
        else:
            self._set_model_gpus(cfgs.gpus)

    def _build_optimizers(self, cfgs):
        # loss, optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfgs.lr, weight_decay=cfgs.weight_decay)
        self.scheduler_type = cfgs.scheduler
        if self.scheduler_type == ScheduleMethods.step:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, cfgs.lr_decay_steps, gamma=cfgs.lr_decay)
        elif self.scheduler_type == ScheduleMethods.cosine:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, eta_min=cfgs.lr_min,
                                                                                  T_0=cfgs.lr_t0, T_mult=cfgs.lr_tm)
        else:
            raise ValueError("the current scheduler is not defined")

        if self.snapshot and not cfgs.only_model:
            self.load_learning_parameters(self.state_dict)

    def _set_model_gpus(self, cfg):
        # self.current_rank = 0  # global rank
        # cfg.local_rank = os.environ['LOCAL_RANK']
        if self.distributed:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            print("os world size: {}, local_rank: {}, rank: {}".format(world_size, local_rank, rank))

            # this will make all .cuda() calls work properly
            torch.cuda.set_device(cfg.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=5000))
            # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
            world_size = dist.get_world_size()
            self.current_rank = dist.get_rank()
            # self.logger.info\
            print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                  % (self.current_rank, world_size))

            # synchronizes all the threads to reach this point before moving on
            dist.barrier()
        else:
            # self.logger.info\
            print('Training with a single process on 1 GPUs.')
        assert self.current_rank >= 0, "rank is < 0"

        # if cfg.local_rank == 0:
        #     self.logger.info(
        #         f'Model created, param count:{sum([m.numel() for m in self.model.parameters()])}')

        # move model to GPU, enable channels last layout if set
        if self.distributed:
            self.model.cuda()
        else:
            self.model.to(self.device)

        if cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.distributed and cfg.sync_bn:
            assert not cfg.split_bn
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if cfg.local_rank == 0:
                print(
                    'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                    'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

        # setup distributed training
        if self.distributed:
            if cfg.local_rank == 0:
                print("Using native Torch DistributedDataParallel.")
            self.model = DDP(self.model, device_ids=[cfg.local_rank],
                             broadcast_buffers=not cfg.no_ddp_bb,
                             find_unused_parameters=True)
            # NOTE: EMA model does not need to be wrapped by DDP

        # # setup exponential moving average of model weights, SWA could be used here too
        # model_ema = None
        # if args.model_ema:
        #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        #     model_ema = ModelEmaV2(
        #         self.model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)

    def load_snapshot(self, snapshot):
        """
        Load the parameters of the model and the training class
        Args:
            snapshot: the complete path to the snapshot file
        """
        print('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=torch.device(self.device))

        # Load model
        model_dict = state_dict['state_dict']
        self.model.load_state_dict(model_dict, strict=False)

        # log missing keys and unexpected keys
        snapshot_keys = set(model_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if len(missing_keys) > 0:
            warn('Missing keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            warn('Unexpected keys: {}'.format(unexpected_keys))
        print('Model has been loaded.')
        return state_dict

    def load_learning_parameters(self, state_dict):
        # Load other attributes
        if 'epoch' in state_dict:
            self.epoch = state_dict['epoch']
            print('Epoch has been loaded: {}.'.format(self.epoch))
        if 'iteration' in state_dict:
            self.iteration = state_dict['iteration']
            print('Iteration has been loaded: {}.'.format(self.iteration))
        if 'optimizer' in state_dict and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state_dict['optimizer'])
                print('Optimizer has been loaded.')
            except:
                print("doesn't load optimizer")
        if 'scheduler' in state_dict and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state_dict['scheduler'])
                print('Scheduler has been loaded.')
            except:
                print("doesn't load scheduler")

    def save_snapshot(self, filename, all=True):
        """
        save the snapshot of the model and other training parameters
        Args:
            filename: the output filename that is the full directory
        """
        if self.distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        # save model
        state_dict = {'state_dict': model_state_dict}
        torch.save(state_dict, filename)
        # print('Model saved to "{}"'.format(filename))

        # save snapshot
        if all:
            state_dict['epoch'] = self.epoch
            state_dict['iteration'] = self.iteration
            snapshot_filename = osp.join(self.output_dir, str(self.name) + '_snapshot.tar')
            state_dict['optimizer'] = self.optimizer.state_dict()
            if self.scheduler is not None:
                state_dict['scheduler'] = self.scheduler.state_dict()
            torch.save(state_dict, snapshot_filename)
        # print('Snapshot saved to "{}"'.format(snapshot_filename))

    def set_train_mode(self):
        """
        set the model to the training mode: parameters are differentiable
        """
        self.training = True
        self.model.train()
        torch.set_grad_enabled(True)

    def set_eval_mode(self):
        """
        set the model to the evaluation mode: parameters are not differentiable
        """
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)
