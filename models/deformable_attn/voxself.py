import numpy as np
import torch
from torch import nn

from models.voxformer import PerceptionTransformer
from models.voxformer.header import Header


class VoxSelf(nn.Module):
    def __init__(self, self_transformer, vox_size, embed_dims):
        super().__init__()
        self.self_transformer = PerceptionTransformer(
            num_cams=self_transformer.num_cams,
            two_stage_num_proposals=300,
            encoder=self_transformer.encoder,
            embed_dims=self_transformer.embed_dims,
            rotate_prev_bev=self_transformer.rotate_prev_bev,
            use_shift=self_transformer.use_shift,
            use_cams_embeds=True,
            rotate_center=[100, 100])  # build_transformer()
        self.header = Header(self.n_classes, nn.BatchNorm3d, feature=self.embed_dims)

        self.bev_h = vox_size[0]
        self.bev_w = vox_size[1]
        self.bev_z = vox_size[2]
        self.embed_dims = embed_dims
        self.bev_embed = nn.Embedding((self.bev_h) * (self.bev_w) * (self.bev_z), self.embed_dims)

    def get_ref_3d(self):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
            ref_3d (Array): 3D reference points
        """
        scene_size = (51.2, 51.2, 6.4)
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = self.real_h / self.bev_h

        vol_bnds = np.zeros((3, 2))
        vol_bnds[:, 0] = vox_origin
        vol_bnds[:, 1] = vox_origin + np.array(scene_size)

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

    def forward(self, mlvl_feats, lidar2img, vox_feats_flatten):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        bev_queries = self.bev_embed.weight.to(dtype)  # [128*128*16, dim]

        vox_coords, ref_3d = self.get_ref_3d()
        # proposal = proposal.reshape(bs, self.bev_h, self.bev_w, self.bev_z)[0]
        # unmasked_idx = np.asarray(np.where(proposal.reshape(-1) > 0)).astype(np.int32)
        # masked_idx = np.asarray(np.where(proposal.reshape(-1) == 0)).astype(np.int32)

        bev_pos_self_attn = self.positional_encoding(
            torch.zeros((bs, 512, 512), device=bev_queries.device).to(dtype)).to(dtype)  # [1, dim, 128*4, 128*4]
        # Complete voxel features by adding mask tokens
        # vox_feats = torch.empty((self.bev_h, self.bev_w, self.bev_z, self.embed_dims), device=bev_queries.device)
        # vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        # vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = seed_feats[0]
        # vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = self.mask_embed.weight.view(1, self.embed_dims).expand(
        #     masked_idx.shape[1], self.embed_dims).to(dtype)

        # Diffuse voxel features by deformable self attention
        vox_feats_diff = self.self_transformer.diffuse_vox_features(
            mlvl_feats,
            vox_feats_flatten,
            512,
            512,
            lidar2img=lidar2img,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=None,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos_self_attn,
            # img_metas=img_metas,
            prev_bev=None,
        )
        vox_feats_diff = vox_feats_diff.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims)

        out = self.header(vox_feats_diff.permute(3, 0, 1, 2).unsqueeze(0))
        return out