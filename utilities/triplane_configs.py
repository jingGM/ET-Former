import copy

import numpy as np
from easydict import EasyDict as edict

from utilities.kitti_configs import KiTTi_Class_Names


class SelfDATConfig:
    def __init__(self, fmap_size=[(128, 128)], window_size=(7, 7), dims=[128, 128], depths=[2], stage_spec=[["N", "D"]],
                 n_groups=2, use_pe=True, sr_ratio=[8], heads=[4], strides=[4], offset_range_factor=-1,
                 no_offs=False, fixed_pe=False, dwc_pes=False, use_lpu=True, nat_ksize=7, ksize=10,
                 attn_drop_rate=0.0, proj_drop=0.0, drop_path_rate=[0.0, 0.0], expansion=4, use_dwc_mlp=True):
        self.fmap_size = fmap_size
        self.window_size = window_size
        self.dims = dims
        self.depths = depths
        self.stage_spec = stage_spec
        self.n_groups = n_groups
        self.use_pe = use_pe
        self.sr_ratio = sr_ratio
        self.heads = heads
        self.strides = strides  # decide number of key and values, num_q / strides
        self.offset_range_factor = offset_range_factor
        self.no_offs = no_offs
        self.fixed_pe = fixed_pe
        self.dwc_pes = dwc_pes
        self.use_lpu = use_lpu
        self.use_dwc_mlp = use_dwc_mlp
        self.nat_ksize = nat_ksize
        self.ksize = ksize
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop = proj_drop
        self.drop_path_rate = drop_path_rate
        self.expansion = expansion


class CrossDATTPVConfig:
    def __init__(self, vox_size=[128, 128, 16], image_size=[370, 1220], reference_ratio=4, heads_num=4, num_levels=4,
                 in_dim=128, out_dim=128, kernel_size=5):
        self.vox_size = vox_size
        self.reference_ratio = reference_ratio
        self.reference_num = np.array(np.array(vox_size) / reference_ratio, dtype=int)
        self.image_size = image_size
        self.heads_num = heads_num
        self.num_levels = num_levels

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size


class MergeType:
    concat = "concat"
    sum = "sum"


class CrossDATConfig:
    def __init__(self, vox_size=[128, 128, 16], image_size=[370, 1220], reference_ratio=8, heads_num=8, num_levels=4,
                 in_dim=128, out_dim=128, kernel_size=5):
        self.vox_size = np.array(vox_size)
        self.reference_ratio = reference_ratio
        self.reference_num = np.array(np.array(vox_size) / reference_ratio, dtype=int)
        self.image_size = image_size
        self.heads_num = heads_num
        self.num_levels = num_levels

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size


TriplaneConfig = edict()
TriplaneConfig.voxel_size = [128, 128, 16]
TriplaneConfig.pe_num_emb = [128, 128, 16]
TriplaneConfig.pe_num_fts = 128
TriplaneConfig.merge_type = MergeType.concat

TriplaneConfig.self_hw = SelfDATConfig()
TriplaneConfig.self_wd = SelfDATConfig(fmap_size=[(128, 16)])
TriplaneConfig.self_hd = SelfDATConfig(fmap_size=[(128, 16)])

TriplaneConfig.cross = CrossDATConfig(vox_size=TriplaneConfig.voxel_size, heads_num=8)
TriplaneConfig.residual = True
TriplaneConfig.double = False
TriplaneConfig.use_position = True
TriplaneConfig.preprocess_fts = False

TriplaneConfig.reduce_dim = 32

TriplaneConfig.hidden_dims = 256
TriplaneConfig.n_classes = len(KiTTi_Class_Names)
