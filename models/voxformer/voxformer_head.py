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

import os
import torch
import numpy as np
import torch.nn as nn

from models.voxformer.transformer import PerceptionTransformer
from models.voxformer.header import Header
from utilities.configs import DataDict, ModelTypes


# from models.voxformer.utils.ssc_loss import sem_scal_loss, KL_sep, geo_scal_loss, CE_ssc_loss


class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat((x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)),
                        dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str


# @HEADS.register_module()
class VoxFormerHead(nn.Module):
    def __init__(
            self, vox_size, real_size, n_classes,
            cross_transformer,
            self_transformer,
            positional_encoding,
            embed_dims,
            save_flag=False,
    ):
        super().__init__()
        self.bev_h = vox_size[0]
        self.bev_w = vox_size[1]
        self.bev_z = vox_size[2]
        self.real_h = real_size[0]
        self.real_w = real_size[1]
        self.scene_size = real_size
        self.n_classes = n_classes
        self.embed_dims = embed_dims

        self.bev_embed = nn.Embedding(self.bev_h * self.bev_w * self.bev_z, self.embed_dims)
        self.mask_embed = nn.Embedding(1, self.embed_dims)
        self.positional_encoding = LearnedPositionalEncoding(num_feats=positional_encoding.num_feats,
                                                             row_num_embed=positional_encoding.row_num_embed,
                                                             col_num_embed=positional_encoding.col_num_embed)
        self.cross_transformer = PerceptionTransformer(
            num_cams=cross_transformer.num_cams,
            two_stage_num_proposals=300,
            encoder=cross_transformer.encoder,
            embed_dims=cross_transformer.embed_dims,
            rotate_prev_bev=cross_transformer.rotate_prev_bev,
            use_shift=cross_transformer.use_shift,
            use_cams_embeds=True,
            rotate_center=[100, 100])  # build_transformer()

    def get_ref_3d(self, vox_origin):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
            ref_3d (Array): 3D reference points
        """
        voxel_size = self.real_h / self.bev_h

        vol_bnds = np.zeros((3, 2))
        vol_bnds[:, 0] = vox_origin
        vol_bnds[:, 1] = vox_origin + np.array(self.scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0] * vol_dim[1] * vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1), idx],
                                    axis=0).astype(int).T

        # Normalize the voxels centroids in lidar cooridnates
        ref_3d = np.concatenate([(xv.reshape(1, -1) + 0.5) / self.bev_h, (yv.reshape(1, -1) + 0.5) / self.bev_w,
                                 (zv.reshape(1, -1) + 0.5) / self.bev_z, ], axis=0).astype(np.float64).T

        return vox_coords, ref_3d

    def forward(self, mlvl_feats, proposal, lidar2img, vox_origin, pc_range):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information such as camera intrinsics.
            # target: Semantic completion ground truth.
        Returns:
            ssc_logit (Tensor): Outputs from the segmentation head.
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embed.weight.to(dtype)  #[128*128*16, dim]

        # Generate bev postional embeddings for cross and self attention
        bev_pos_cross_attn = self.positional_encoding(
            torch.zeros((bs, 512, 512), device=bev_queries.device).to(dtype)).to(dtype)  # [1, dim, 128*4, 128*4]

        # Load query proposals
        # proposal = img_metas[0]['proposal'].reshape(self.bev_h, self.bev_w, self.bev_z)
        proposal = proposal.reshape(bs, self.bev_h, self.bev_w, self.bev_z)[0]
        unmasked_idx = np.asarray(np.where(proposal.reshape(-1) > 0)).astype(np.int32)
        masked_idx = np.asarray(np.where(proposal.reshape(-1) == 0)).astype(np.int32)
        vox_coords, ref_3d = self.get_ref_3d(vox_origin=vox_origin)

        # Compute seed features of query proposals by deformable cross attention
        seed_feats = self.cross_transformer.get_vox_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            lidar2img=lidar2img,
            pc_range=pc_range,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos_cross_attn,
            # img_metas=img_metas,
            prev_bev=None,
        )

        return seed_feats
