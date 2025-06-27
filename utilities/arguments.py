import argparse
import copy

import numpy as np
import torch
from utilities.configs import TrainingConfig, DatasetTypes, ModelTypes, KiTTi_Class_Names, \
    KiTTi_Class_Weights, ScheduleMethods, Stage1UNetConfig, QueryTypes
from utilities.triplane_configs import TriplaneConfig, SelfDATConfig, CrossDATConfig


def get_args():
    parser = argparse.ArgumentParser(description='etformer')
    parser.add_argument('--name', type=str, help='name of the experiment', default="")

    # data
    parser.add_argument('--data_cfg', type=str, help='configuration file of dataset',
                        default="data/semantic_kitti/semantic-kitti.yaml")
    parser.add_argument('--data_root', type=str, help='root of all datasets', default="")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--workers', type=int, default=16, help="the worker number in the dataloader")

    # model
    parser.add_argument('--model_type', type=int, default=0, help="1: triplane, 2: cvae, 3: stage 1")
    parser.add_argument('--snapshot', type=str, help='snapshot', default="")
    parser.add_argument('--only_load_model', action='store_true', default=False, help='only load model to continue training')
    parser.add_argument('--distribute_evaluation', action='store_true', default=False, help='use distributed data loader for evaluation')
    parser.add_argument('--evaluation_freq', type=int, default=2, help="evaluation frequency")
    parser.add_argument('--no_log', action='store_true', default=False, help='no_log')
    parser.add_argument('--generate_data', action='store_true', default=False, help='generate_data')
    parser.add_argument('--cvae_ratio', type=float, default=0.1, help='no cvae loss ratio')
    parser.add_argument('--loss_alpha', type=float, default=0.54, help='alpha value of binary cross entropy loss for occupied')

    parser.add_argument('--scheduler', type=int, default=1, help="0: step; 1: cosine 1e-4, 2:cosine 5e-4, 3:cosine 1e-3")

    parser.add_argument('--wandb_api', type=str, help='w&b api', default="")
    parser.add_argument('--wandb_proj', type=str, help='the project name of w&b', default="")

    # GPUs
    # parser.add_argument('--distributed', action='store_true', default=False, help='if using multiple GPUs')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument('--sync-bn', action='store_true', default=False,
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--split-bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')
    parser.add_argument('--device', type=int, default=-1, help="the gpu id")
    parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                        help='Force broadcast buffers for native DDP to off.')
    return parser.parse_args()


def get_configuration():
    args = get_args()
    cfg = TrainingConfig

    #########################################
    # data configurations
    #########################################
    cfg.data.config_dir = args.data_cfg
    cfg.data.data_root = args.data_root
    cfg.data.workers = args.workers
    cfg.data.batch_size = args.batch_size

    cfg.data.data_type = DatasetTypes.kitti

    cfg.loss.n_classes = len(KiTTi_Class_Names)
    cfg.loss.class_names = KiTTi_Class_Names
    cfg.loss.class_weights = KiTTi_Class_Weights

    vox_size = np.array([256, 256, 32]) / 2
    cfg.model.triplane.voxel_size = cfg.model.triplane.pe_num_emb = vox_size.astype(int)
    cfg.model.triplane.self_hw = SelfDATConfig(
        fmap_size=[(cfg.model.triplane.voxel_size[0], cfg.model.triplane.voxel_size[1])])
    cfg.model.triplane.self_wd = SelfDATConfig(
        fmap_size=[(cfg.model.triplane.voxel_size[1], cfg.model.triplane.voxel_size[2])])
    cfg.model.triplane.self_hd = SelfDATConfig(
        fmap_size=[(cfg.model.triplane.voxel_size[0], cfg.model.triplane.voxel_size[2])])
    cfg.model.triplane.cross = CrossDATConfig(
        vox_size=cfg.model.triplane.voxel_size, heads_num=8,
        image_size=[370, 1220]
    )
    cfg.model.triplane.n_classes = len(KiTTi_Class_Names)

    cfg.model.vox_former.pts_bbox_head.vox_size = vox_size.astype(int)
    cfg.model.vox_former.pts_bbox_head.real_size = np.array([51.2, 51.2, 6.4])
    cfg.model.vox_former.pts_bbox_head.n_classes = cfg.loss.n_classes
    cfg.model.vox_former.pts_bbox_head.cross_transformer.encoder.image_size = cfg.model.vox_former.pts_bbox_head.self_transformer.encoder.image_size = [
        370, 1220]

    #########################################
    # model configurations
    #########################################
    if args.model_type == 1:
        cfg.model.type = ModelTypes.triplane
    elif args.model_type == 2:
        cfg.model.type = ModelTypes.cvae
        cfg.model.decoder = copy.deepcopy(TriplaneConfig)
    elif args.model_type == 3:
        cfg.model.type = cfg.model.baseline_type = ModelTypes.query

        # args.input_scale
        query_ref_ratio = 8
        cfg.model.stage1.triplane.n_classes = 2
        cfg.model.stage1.triplane.voxel_size = (np.array([256, 256, 32]) / cfg.data.scale).astype(int)
        cfg.model.stage1.triplane.pe_num_emb = cfg.model.stage1.triplane.voxel_size
        cfg.model.stage1.triplane.self_hw = SelfDATConfig(fmap_size=[(128, 128)], sr_ratio=[query_ref_ratio])
        cfg.model.stage1.triplane.self_wd = SelfDATConfig(fmap_size=[(128, 16)], sr_ratio=[query_ref_ratio])
        cfg.model.stage1.triplane.self_hd = SelfDATConfig(fmap_size=[(128, 16)], sr_ratio=[query_ref_ratio])
        cfg.model.stage1.triplane.cross = CrossDATConfig(vox_size=[128, 128, 16], heads_num=8,
                                                         reference_ratio=query_ref_ratio)
    else:
        raise Exception("the model type {} is not defined".format(args.model_type))

    cfg.model.triplane.preprocess_fts = cfg.model.stage1.triplane.preprocess_fts = False
    cfg.model.triplane.residual = True
    cfg.model.triplane.double = cfg.model.stage1.triplane.double = False

    #########################################
    # training configurations
    #########################################
    cfg.distribute_evaluation = args.distribute_evaluation
    cfg.only_model = args.only_load_model
    cfg.name = args.name
    cfg.snapshot = args.snapshot
    cfg.evaluation_freq = args.evaluation_freq
    cfg.no_log = args.no_log
    cfg.generate_data = args.generate_data
    cfg.loss.use_occ = True
    cfg.loss.cvae_ratio = args.cvae_ratio
    cfg.loss.alpha = args.loss_alpha

    if args.scheduler == 0:
        cfg.scheduler = ScheduleMethods.step
        cfg.lr = 1e-4
        cfg.weight_decay = 1e-5
        cfg.max_epoch = 100
        cfg.lr_decay = 0.8
        cfg.lr_decay_steps = 3
    elif args.scheduler == 1:
        cfg.scheduler = ScheduleMethods.cosine
        cfg.max_epoch = 90
        cfg.lr = 1e-4
        cfg.weight_decay = 1e-4
        # for cosine scheduler
        cfg.lr_t0 = 1
        cfg.lr_tm = 9
        cfg.lr_min = 1e-8
    elif args.scheduler == 2:
        cfg.scheduler = ScheduleMethods.cosine
        cfg.max_epoch = 156
        cfg.lr = 2e-4
        cfg.weight_decay = 1e-4
        # for cosine scheduler
        cfg.lr_t0 = 1
        cfg.lr_tm = 12
        cfg.lr_min = 1e-8
    elif args.scheduler == 3:
        cfg.scheduler = ScheduleMethods.cosine
        cfg.max_epoch = 108
        cfg.lr = 1e-3
        cfg.weight_decay = 1e-4
        # for cosine scheduler
        cfg.lr_t0 = 10
        cfg.lr_tm = 10
        cfg.lr_min = 1e-8
    else:
        raise Exception("the scheduler type {} is not defined".format(args.scheduler))

    # Devices
    # TrainingConfig.data.distributed = args.distributed
    if args.device >= 0:
        cfg.gpus.device = "cuda:{}".format(args.device)
    elif args.device == -1:
        cfg.gpus.device = "cuda"
        print("------------------- CUDA: ", torch.cuda.is_available(), "-----------------------")
    else:
        cfg.gpus.device = "cpu"
    # print(" ------ local rank: ", os.environ['LOCAL_RANK'])
    cfg.gpus.channels_last = args.channels_last
    cfg.gpus.local_rank = args.local_rank
    cfg.gpus.sync_bn = args.sync_bn
    cfg.gpus.split_bn = args.split_bn
    cfg.gpus.no_ddp_bb = args.no_ddp_bb

    cfg.wandb_api = args.wandb_api
    cfg.wandb_proj = args.wandb_proj

    return cfg