import copy

import numpy as np
from easydict import EasyDict as edict

from utilities.kitti_configs import KiTTi_Class_Names


class TransformerType:
    self = "self"
    cross = "cross"


ImgNeckConfig = edict()
ImgNeckConfig.in_channels = [1024]
ImgNeckConfig.out_channels = 128
ImgNeckConfig.start_level = 0
ImgNeckConfig.add_extra_convs = 'on_output'
ImgNeckConfig.num_outs = 1
ImgNeckConfig.relu_before_extra_convs = True

ResNetConfig = edict()
ResNetConfig.depth = 50
ResNetConfig.num_stages = 4
ResNetConfig.out_indices = (2,)
ResNetConfig.frozen_stages = 1
ResNetConfig.norm_cfg = dict(type='BN', requires_grad=False)
ResNetConfig.norm_eval = True
ResNetConfig.style = 'pytorch'
ResNetConfig.stage_with_dcn = (False, False, False, False)

FFNConfig = edict()
FFNConfig.embed_dims = 128
FFNConfig.feedforward_channels = 128 * 2
FFNConfig.num_fcs = 2
FFNConfig.ffn_drop = 0.1
FFNConfig.act_cfg = dict(type='ReLU', inplace=True)

MSDeformableAttention3DConfig = edict()
MSDeformableAttention3DConfig.embed_dims = 128
MSDeformableAttention3DConfig.num_points = 8
MSDeformableAttention3DConfig.num_levels = 1

DeformCrossAttentionConfig = edict()
DeformCrossAttentionConfig.num_cams = 1
DeformCrossAttentionConfig.embed_dims = 128
DeformCrossAttentionConfig.deformable_attention = MSDeformableAttention3DConfig

CrossVoxFormerLayerConfig = edict()
CrossVoxFormerLayerConfig.type = TransformerType.cross
CrossVoxFormerLayerConfig.attn_cfgs = [DeformCrossAttentionConfig]
CrossVoxFormerLayerConfig.ffn_cfgs = FFNConfig
CrossVoxFormerLayerConfig.operation_order = ('cross_attn', 'norm', 'ffn', 'norm')

CrossVoxFormerEncoderConfig = edict()
CrossVoxFormerEncoderConfig.num_layers = 3
CrossVoxFormerEncoderConfig.image_size = [370, 1220]
# CrossVoxFormerEncoderConfig.pc_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
CrossVoxFormerEncoderConfig.num_points_in_pillar = 8
CrossVoxFormerEncoderConfig.return_intermediate = False
CrossVoxFormerEncoderConfig.transformerlayers = CrossVoxFormerLayerConfig

CrossPerceptionTransformerConfig = edict()
CrossPerceptionTransformerConfig.rotate_prev_bev = True
CrossPerceptionTransformerConfig.use_shift = True
CrossPerceptionTransformerConfig.embed_dims = 128
CrossPerceptionTransformerConfig.num_cams = 1
CrossPerceptionTransformerConfig.encoder = CrossVoxFormerEncoderConfig

DeformSelfAttentionConfig = edict()
DeformSelfAttentionConfig.embed_dims = 128
DeformSelfAttentionConfig.num_levels = 1
DeformSelfAttentionConfig.num_points = 8

SelfVoxFormerLayerConfig = edict()
SelfVoxFormerLayerConfig.type = TransformerType.self
SelfVoxFormerLayerConfig.attn_cfgs = [DeformSelfAttentionConfig]
SelfVoxFormerLayerConfig.ffn_cfgs = FFNConfig
SelfVoxFormerLayerConfig.operation_order = ('self_attn', 'norm', 'ffn', 'norm')

SelfVoxFormerEncoderConfig = copy.deepcopy(CrossVoxFormerEncoderConfig)
SelfVoxFormerEncoderConfig.num_layers = 2
SelfVoxFormerEncoderConfig.transformerlayers = SelfVoxFormerLayerConfig

SelfPerceptionTransformerConfig = copy.deepcopy(CrossPerceptionTransformerConfig)
SelfPerceptionTransformerConfig.encoder = SelfVoxFormerEncoderConfig

LearnedPositionalEncodingConfig = edict()
LearnedPositionalEncodingConfig.num_feats = 128 // 2
LearnedPositionalEncodingConfig.row_num_embed = 512
LearnedPositionalEncodingConfig.col_num_embed = 512

VoxFormerHead = edict()
VoxFormerHead.vox_size = [128, 128, 16]
VoxFormerHead.real_size = [51.2, 51.2, 6.4]
VoxFormerHead.embed_dims = 128
VoxFormerHead.n_classes = len(KiTTi_Class_Names)
VoxFormerHead.cross_transformer = CrossPerceptionTransformerConfig
VoxFormerHead.self_transformer = SelfPerceptionTransformerConfig
VoxFormerHead.positional_encoding = LearnedPositionalEncodingConfig

VoxFormerConfig = edict()
VoxFormerConfig.img_backbone = ResNetConfig
VoxFormerConfig.img_neck = ImgNeckConfig
VoxFormerConfig.pts_bbox_head = VoxFormerHead
