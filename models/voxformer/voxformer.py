# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
from models.voxformer.voxformer_head import VoxFormerHead
from models.voxformer.image_necks import FPN
from models.voxformer.resnet import ResNet
from utilities.configs import DataDict, ModelTypes
from models.commons import BaseModel
from utilities.functions import to_device


class VoxFormer(BaseModel):
    def __init__(self, config, data_type, model_type, device="cpu"):
        super().__init__(config=config, data_type=data_type, device=device, model_type=model_type)
        self.img_backbone = ResNet(depth=config.img_backbone.depth, num_stages=config.img_backbone.num_stages,
                                   out_indices=config.img_backbone.out_indices,
                                   frozen_stages=config.img_backbone.frozen_stages,
                                   norm_cfg=config.img_backbone.norm_cfg, norm_eval=config.img_backbone.norm_eval,
                                   style=config.img_backbone.style, stage_with_dcn=config.img_backbone.stage_with_dcn)
        self.img_neck = FPN(in_channels=config.img_neck.in_channels, out_channels=config.img_neck.out_channels,
                            start_level=config.img_neck.start_level, add_extra_convs=config.img_neck.add_extra_convs,
                            num_outs=config.img_neck.num_outs,
                            relu_before_extra_convs=config.img_neck.relu_before_extra_convs)

        self.pts_bbox_head = VoxFormerHead(vox_size=config.pts_bbox_head.vox_size, real_size=config.pts_bbox_head.real_size,
                                           cross_transformer=config.pts_bbox_head.cross_transformer,
                                           n_classes=config.pts_bbox_head.n_classes,
                                           self_transformer=config.pts_bbox_head.self_transformer,
                                           positional_encoding=config.pts_bbox_head.positional_encoding,
                                           embed_dims=config.pts_bbox_head.embed_dims
                                           )

    def extract_img_feat(self, img, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img.dim() == 5 and img.size(0) == 1:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        # if self.with_img_neck:
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    def forward(self, data_dict):
        img_feats = self.extract_img_feat(img=to_device(data_dict[DataDict.img], device=self.device))

        outs = self.pts_bbox_head(mlvl_feats=img_feats, proposal=data_dict[DataDict.proposal].cpu().numpy(),
                                  lidar2img=data_dict[DataDict.w2i], pc_range=data_dict[DataDict.pc_range].cpu().numpy(),
                                  vox_origin=data_dict[DataDict.vox_origin].cpu().numpy(),)
        return outs, img_feats
