import copy
import math
from os.path import join
import numpy as np
import yaml
from easydict import EasyDict as edict

from utilities.kitti_configs import DataDict, DataDict, KiTTi_Class_Weights, KiTTi_Class_Names, semantic_kitti_class_frequencies
from utilities.triplane_configs import TriplaneConfig
from utilities.voxformer_configs import VoxFormerConfig, ResNetConfig, ImgNeckConfig


#########################################################################
# Configuration of Dataset and Data Loader
#########################################################################
class DatasetUsages:
    train = "train"
    val = "valid"
    test = "test"
    sanity = "sanity"


class DatasetTypes:
    kitti = "kitti"
    nyu = "nyu"


class QueryTypes:
    vox_output = "vox_output"
    stereo_input = "stereo_input"
    mono_input = "mono_input"


DatasetConfig = edict()  # Configuration of data loaders
DatasetConfig.config_dir = ""
DatasetConfig.data_root = ""
DatasetConfig.data_type = DatasetTypes.kitti
DatasetConfig.batch_size = 1
DatasetConfig.workers = 8
DatasetConfig.distributed = False
DatasetConfig.scale = 1
DatasetConfig.our_query = False


#########################################################################
# Configuration of Models
#########################################################################
class ModelTypes:
    triplane = "triplane"
    cvae = "cvae"
    query = "query"
    debug = "debug"

FFNConfig = edict()
FFNConfig.embed_dims = 128
FFNConfig.feedforward_channels = 128 * 2
FFNConfig.num_fcs = 2
FFNConfig.ffn_drop = 0.1
FFNConfig.act_cfg = dict(type='ReLU', inplace=True)

Stage1UNetConfig = edict()
Stage1UNetConfig.class_num = 2
Stage1UNetConfig.input_dimensions = [256, 32, 256]
Stage1UNetConfig.out_scale = "1_2"
Stage1UNetConfig.img_backbone = ResNetConfig
Stage1UNetConfig.img_neck = ImgNeckConfig
Stage1UNetConfig.triplane = copy.deepcopy(TriplaneConfig)
Stage1UNetConfig.hidden_dims = 256  #64
Stage1UNetConfig.scale = 1

ModelConfig = edict()
ModelConfig.type = ModelTypes.triplane

ModelConfig.vox_former = VoxFormerConfig
ModelConfig.triplane = TriplaneConfig
ModelConfig.decoder = copy.deepcopy(TriplaneConfig)
ModelConfig.stage1 = Stage1UNetConfig
ModelConfig.cvae_in_dim = 128
ModelConfig.stage = 2

#########################################################################
# Configuration of Loss
#########################################################################
class Hausdorff:
    average = "average"
    max = "max"


class LossNames:
    multilabel_loss = "multilabel_loss"
    ce_ssc_loss = "CE_ssc_loss"
    sem_scal = "sem_scal"
    geo_scal = "geo_scal"
    frustum_loss = "frustum_loss"
    cvae = "cvae"
    sc_level_1 = "sc_level_1"

    loss = "loss"


class EvaluationNames:
    precision = "Precision"
    recall = "Recall"
    iou_occ = "IoU"
    iou_ssc = "IoU_ssc"
    iou_ssc_mean = "mIoU"


LossConfig = edict()
LossConfig.ssc_mean = False
LossConfig.class_weights = KiTTi_Class_Weights
LossConfig.class_names = KiTTi_Class_Names
LossConfig.n_classes = len(KiTTi_Class_Names)
LossConfig.use_occ = True
LossConfig.alpha = 0.54
LossConfig.binary = False
LossConfig.cvae_ratio = 0.01
LossConfig.class_frequencies_level1 = np.array([5.41773033e09, 4.03113667e08])


#########################################################################
# Configuration of Training
#########################################################################
class ScheduleMethods:
    step = "step"
    cosine = "cosine"


class LogNames:
    step_time = "step_time"
    lr = "learning_rate"


class LogTypes:
    train = "train"
    others = "evaluation"


TrainingConfig = edict()
TrainingConfig.name = ""
TrainingConfig.distribute_evaluation = False
TrainingConfig.only_model = False
TrainingConfig.output_dir = "./results"
TrainingConfig.snapshot = "./pretrained.pth.tar"
TrainingConfig.max_epoch = 100
TrainingConfig.evaluation_freq = 1
TrainingConfig.scheduler = ScheduleMethods.cosine
TrainingConfig.lr = 1e-5
TrainingConfig.weight_decay = 1e-4
# for cosine scheduler
TrainingConfig.lr_t0 = 1
TrainingConfig.lr_tm = 10
TrainingConfig.lr_min = 1e-8

TrainingConfig.gpus = edict()
TrainingConfig.gpus.channels_last = False
TrainingConfig.gpus.local_rank = 1
TrainingConfig.gpus.sync_bn = False
TrainingConfig.gpus.no_ddp_bb = False
TrainingConfig.gpus.device = "cuda:0"

TrainingConfig.data = DatasetConfig
TrainingConfig.model = ModelConfig
TrainingConfig.loss = LossConfig
TrainingConfig.no_log = False
TrainingConfig.generate_data = False

TrainingConfig.wandb_api = ""
TrainingConfig.wandb_proj = ""
