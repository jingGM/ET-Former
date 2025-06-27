# from mmcv.cnn import xavier_init
from torch import nn
import torch
import numpy as np
from torch.nn.functional import normalize

from models.commons import BaseModel
from models.triplane.cross_attention import CrossDeformableAttention
from utilities.functions import to_device
from models.triplane.self_attention import SelfDeformableAttention
from utilities.triplane_configs import MergeType


class LearnedPositionalEncoding(nn.Module):
    def __init__(self,
                 num_feats=128,
                 num_embed=(50, 50, 50), sparse_embedding=True):
        super(LearnedPositionalEncoding, self).__init__()
        self.h_embed = nn.Embedding(num_embed[0], num_feats)
        self.w_embed = nn.Embedding(num_embed[1], num_feats)
        self.d_embed = nn.Embedding(num_embed[2], num_feats)

        self.sparse_embedding = sparse_embedding
        if self.sparse_embedding:
            self.linear = nn.Sequential(nn.Linear(3 * num_feats, 2 * num_feats), nn.GELU(),
                                        nn.Linear(2 * num_feats, num_feats), nn.GELU(), nn.LayerNorm(num_feats))
        # self.num_feats = num_feats
        # self.row_num_embed = row_num_embed
        # self.col_num_embed = col_num_embed

    def forward(self, poses):
        # b, n, c = poses.shape
        if self.sparse_embedding:
            h = self.h_embed(poses[0])
            w = self.w_embed(poses[1])
            d = self.d_embed(poses[2])
            pos = torch.cat((h, w, d), dim=-1)
            comb = self.linear(pos)

            h_out = normalize(comb * h + h, dim=-1)
            w_out = normalize(comb * w + w, dim=-1)
            d_out = normalize(comb * d + d, dim=-1)
        else:
            h_out = self.h_embed(poses[0])
            w_out = self.w_embed(poses[1])
            d_out = self.d_embed(poses[2])

        return h_out, w_out, d_out


class Triplane(BaseModel):
    def __init__(self, config, data_type, model_type, device, sparse_embedding=True):
        super().__init__(config, data_type, model_type, device)
        # self.vox_size = config.vox_size
        self.ft_dim = config.pe_num_fts
        self.preprocess_fts = config.preprocess_fts
        if self.preprocess_fts:
            self.process_vox_fts = nn.Sequential(nn.Linear(self.ft_dim, self.ft_dim), nn.LeakyReLU(),
                                                 nn.Linear(self.ft_dim, self.ft_dim))
            self.hw_self_linear = nn.Sequential(nn.Linear(config.self_hw.dims[1], config.self_hw.dims[1]),
                                                nn.LeakyReLU(), nn.Linear(config.self_hw.dims[1], config.self_hw.dims[1]))
            self.hd_self_linear = nn.Sequential(nn.Linear(config.self_hd.dims[1], config.self_hd.dims[1]),
                                                nn.LeakyReLU(), nn.Linear(config.self_hd.dims[1], config.self_hd.dims[1]))
            self.wd_self_linear = nn.Sequential(nn.Linear(config.self_wd.dims[1], config.self_wd.dims[1]),
                                                nn.LeakyReLU(), nn.Linear(config.self_wd.dims[1], config.self_wd.dims[1]))
            self.hw_cross_linear = nn.Sequential(nn.Linear(config.cross.out_dim, config.cross.out_dim), nn.LeakyReLU(),
                                                 nn.Linear(config.cross.out_dim, config.cross.out_dim))
            self.hd_cross_linear = nn.Sequential(nn.Linear(config.cross.out_dim, config.cross.out_dim), nn.LeakyReLU(),
                                                 nn.Linear(config.cross.out_dim, config.cross.out_dim))
            self.wd_cross_linear = nn.Sequential(nn.Linear(config.cross.out_dim, config.cross.out_dim), nn.LeakyReLU(),
                                                 nn.Linear(config.cross.out_dim, config.cross.out_dim))

        self.positional_embedding = LearnedPositionalEncoding(num_feats=config.pe_num_fts, num_embed=config.pe_num_emb,
                                                              sparse_embedding=sparse_embedding)

        if not sparse_embedding:
            self.hw_linear = nn.Sequential(
                nn.Linear(config.pe_num_fts + self.ft_dim, (config.pe_num_fts + self.ft_dim) // 2), nn.LeakyReLU(),
                nn.Linear((config.pe_num_fts + self.ft_dim) // 2, self.ft_dim), nn.GELU()
            )
            self.hd_linear = nn.Sequential(
                nn.Linear(config.pe_num_fts + self.ft_dim, (config.pe_num_fts + self.ft_dim) // 2), nn.LeakyReLU(),
                nn.Linear((config.pe_num_fts + self.ft_dim) // 2, self.ft_dim), nn.GELU()
            )
            self.wd_linear = nn.Sequential(
                nn.Linear(config.pe_num_fts + self.ft_dim, (config.pe_num_fts + self.ft_dim) // 2), nn.LeakyReLU(),
                nn.Linear((config.pe_num_fts + self.ft_dim) // 2, self.ft_dim), nn.GELU()
            )
            # xavier_init(self.hw_linear, distribution='uniform', bias=0.)
            # xavier_init(self.hd_linear, distribution='uniform', bias=0.)
            # xavier_init(self.wd_linear, distribution='uniform', bias=0.)

        self.reference_num = config.cross.reference_num
        self.vox_size = config.voxel_size
        self.ref_pts = self._get_reference_points()

        self.self_plan_hw_att = self._build_self_attentions(config.self_hw)
        self.self_plan_hd_att = self._build_self_attentions(config.self_hd)
        self.self_plan_wd_att = self._build_self_attentions(config.self_wd)
        self.cross_atten = self._build_cross_attention(cross_cfg=config.cross)
        # xavier_init(self.self_plan_hw_att, distribution='uniform', bias=0.)
        # xavier_init(self.self_plan_hd_att, distribution='uniform', bias=0.)
        # xavier_init(self.self_plan_wd_att, distribution='uniform', bias=0.)
        # xavier_init(self.cross_atten, distribution='uniform', bias=0.)

        self.double = config.double
        if self.double:
            self.self_plan_hw_att_2 = self._build_self_attentions(config.self_hw)
            self.self_plan_hd_att_2 = self._build_self_attentions(config.self_hd)
            self.self_plan_wd_att_2 = self._build_self_attentions(config.self_wd)
            self.cross_atten_2 = self._build_cross_attention(cross_cfg=config.cross)
            # xavier_init(self.cross_atten_2, distribution='uniform', bias=0.)
            # xavier_init(self.self_plan_wd_att_2, distribution='uniform', bias=0.)
            # xavier_init(self.self_plan_hd_att_2, distribution='uniform', bias=0.)
            # xavier_init(self.self_plan_hw_att_2, distribution='uniform', bias=0.)

        self.merge_type = config.merge_type
        if self.merge_type == MergeType.concat:
            self.merge = nn.Sequential(nn.Linear(3 * config.cross.out_dim, config.cross.out_dim), nn.LeakyReLU(),
                                       nn.Linear(config.cross.out_dim, config.cross.out_dim), nn.Softsign())
        elif self.merge_type == MergeType.sum:
            self.merge = nn.Sequential(nn.Linear(config.cross.out_dim, config.cross.out_dim), nn.LeakyReLU(),
                                       nn.Linear(config.cross.out_dim, config.cross.out_dim), nn.Softsign())
        else:
            raise Exception("the merge type is not defined")
        # xavier_init(self.merge_type, distribution='uniform', bias=0.)

        self.residual = config.residual
        if self.residual:
            self.residual_net = nn.Sequential(nn.Linear(config.cross.out_dim, config.cross.out_dim), nn.LeakyReLU(),
                                              nn.Linear(config.cross.out_dim, config.cross.out_dim), nn.Softsign())
            # xavier_init(self.residual_net, distribution='uniform', bias=0.)

    @torch.no_grad()
    def _get_reference_points(self):
        x = torch.linspace(0.5, self.vox_size[0] - 0.5, self.reference_num[0]).view(-1, 1, 1).expand(
            self.reference_num[0], self.reference_num[1], self.reference_num[2]) / self.vox_size[0]
        y = torch.linspace(0.5, self.vox_size[1] - 0.5, self.reference_num[1]).view(1, -1, 1).expand(
            self.reference_num[0], self.reference_num[1], self.reference_num[2]) / self.vox_size[1]
        z = torch.linspace(0.5, self.vox_size[2] - 0.5, self.reference_num[2]).view(1, 1, -1).expand(
            self.reference_num[0], self.reference_num[1], self.reference_num[2]) / self.vox_size[2]
        ref_pts = torch.stack((x, y, z), -1)
        return ref_pts

    def _build_self_attentions(self, self_cfg):
        return SelfDeformableAttention(fmap_size=self_cfg.fmap_size[0], window_size=self_cfg.window_size,
                                       dim_in=self_cfg.dims[0], dim_embed=self_cfg.dims[1],
                                       depths=self_cfg.depths[0], stage_spec=self_cfg.stage_spec[0],
                                       n_groups=self_cfg.n_groups, use_pe=self_cfg.use_pe,
                                       sr_ratio=self_cfg.sr_ratio[0], heads=self_cfg.heads[0],
                                       stride=self_cfg.strides[0], offset_range_factor=self_cfg.offset_range_factor,
                                       dwc_pe=self_cfg.dwc_pes, no_off=self_cfg.no_offs,
                                       fixed_pe=self_cfg.fixed_pe, attn_drop=self_cfg.attn_drop_rate,
                                       proj_drop=self_cfg.proj_drop, expansion=self_cfg.expansion,
                                       drop=self_cfg.attn_drop_rate, drop_path_rate=self_cfg.drop_path_rate,
                                       use_dwc_mlp=self_cfg.use_dwc_mlp, ksize=self_cfg.ksize,
                                       nat_ksize=self_cfg.nat_ksize, layer_scale_value=-1, use_lpu=self_cfg.use_lpu,
                                       log_cpb=False)

    def _build_cross_attention(self, cross_cfg):
        return CrossDeformableAttention(vox_size=cross_cfg.vox_size, image_size=cross_cfg.image_size,
                                        reference_num=cross_cfg.reference_num, heads_num=cross_cfg.heads_num,
                                        num_levels=cross_cfg.num_levels,
                                        device=self.device, reference_ratio=cross_cfg.reference_ratio,
                                        in_dim=cross_cfg.in_dim, out_dim=cross_cfg.out_dim,
                                        kernel_size=cross_cfg.kernel_size)

    def _get_plane(self, indices, features, axis=0):
        if axis == 0:
            xy = indices[1:, :].T
        elif axis == 1:
            xy = indices[(0, 2), :].T
        elif axis == 2:
            xy = indices[:2, :].T
        else:
            raise Exception("the axis can only be 0-2")

        unique_xy, inverse_indices = torch.unique(xy, return_inverse=True, dim=0)

        # Initialize a tensor to hold the aggregated features
        aggregated_features = torch.zeros(unique_xy.size(0), features.size(1), dtype=features.dtype, device=self.device)

        # Aggregate feature values based on the inverse indices
        inverse_indices = to_device(inverse_indices, device=self.device)
        aggregated_features.index_add_(0, inverse_indices, features)
        # aggregated_features.scatter_add_(0, inverse_indices.unsqueeze(1).expand_as(features), features)

        # Count the occurrences of each unique (x, y) pair
        # counts = torch.zeros(unique_xy.size(0), dtype=torch.float32, device=self.device)
        # counts.scatter_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=features.dtype))

        # Calculate the average features
        average_features = normalize(aggregated_features, dim=-1)
        return average_features, unique_xy

    def _get_triplane_features(self, proposal=None, queries=None, vox_feats=None):
        """
        args:
        proposa: HxWxD, 0-1
        queries: avalaible queries indices in proposals
        vox_feats: queries features
        """
        if queries is None:
            proposal = proposal[0].reshape(self.vox_size[0], self.vox_size[1], self.vox_size[2])
            queries = torch.from_numpy(np.asarray(np.where(proposal > 0)))
        else:
            queries = queries[0]
        h_features, w_features, d_features = self.positional_embedding(to_device(queries, device=self.device))
        if vox_feats is not None:
            h_features = h_features * vox_feats[0]
            w_features = w_features * vox_feats[0]
            d_features = d_features * vox_feats[0]

        feats_hw, indices_hw = self._get_plane(indices=queries, features=d_features, axis=2)
        feats_hd, indices_hd = self._get_plane(indices=queries, features=w_features, axis=1)
        feats_wd, indices_wd = self._get_plane(indices=queries, features=h_features, axis=0)

        plane_hw = torch.zeros((self.vox_size[0], self.vox_size[1], self.ft_dim), device=self.device)
        plane_hw[indices_hw[:, 0], indices_hw[:, 1]] = feats_hw

        plane_hd = torch.zeros((self.vox_size[0], self.vox_size[2], self.ft_dim), device=self.device)
        plane_hd[indices_hd[:, 0], indices_hd[:, 1]] = feats_hd

        plane_wd = torch.zeros((self.vox_size[1], self.vox_size[2], self.ft_dim), device=self.device)
        plane_wd[indices_wd[:, 0], indices_wd[:, 1]] = feats_wd
        return plane_hw, plane_hd, plane_wd, queries

    def forward(self, img_feats, projection_matrix, pc_range, proposal, queries=None, vox_feats=None):
        assert pc_range.shape[0] == 1, "only support batch_size == 1"
        if vox_feats is not None and self.preprocess_fts:
            vox_feats = self.process_vox_fts(vox_feats)
        plane_hw, plane_hd, plane_wd, indices = self._get_triplane_features(proposal=proposal, vox_feats=vox_feats, queries=queries)  # H, W, C

        hw_self_fts = self.self_plan_hw_att(plane_hw.permute(2, 0, 1).unsqueeze(0))  # B, C, H, W
        hd_self_fts = self.self_plan_hd_att(plane_hd.permute(2, 0, 1).unsqueeze(0))  # B, C, H, W
        wd_self_fts = self.self_plan_wd_att(plane_wd.permute(2, 0, 1).unsqueeze(0))  # B, C, H, W

        if self.preprocess_fts:
            hw_self_fts = self.hw_self_linear(hw_self_fts.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            hd_self_fts = self.hd_self_linear(hd_self_fts.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            wd_self_fts = self.wd_self_linear(wd_self_fts.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        cross_hw, cross_hd, cross_wd = self.cross_atten(hw_self_fts=hw_self_fts, hd_self_fts=hd_self_fts,
                                                        wd_self_fts=wd_self_fts, pc_range=pc_range,
                                                        img_feats=img_feats, projection_matrix=projection_matrix,
                                                        ref_pts=self.ref_pts.clone())  # B, H,W,C
        if self.preprocess_fts:
            cross_hw = self.hw_cross_linear(cross_hw)
            cross_hd = self.hd_cross_linear(cross_hd)
            cross_wd = self.wd_cross_linear(cross_wd)

        if self.double:
            hw_self_fts = self.self_plan_hw_att_2(cross_hw.permute(0, 3, 1, 2))  # B, C, H, W
            hd_self_fts = self.self_plan_hd_att_2(cross_hd.permute(0, 3, 1, 2))  # B, C, H, W
            wd_self_fts = self.self_plan_wd_att_2(cross_wd.permute(0, 3, 1, 2))  # B, C, H, W

            cross_hw, cross_hd, cross_wd = self.cross_atten_2(hw_self_fts=hw_self_fts, hd_self_fts=hd_self_fts,
                                                              wd_self_fts=wd_self_fts,
                                                              img_feats=img_feats, projection_matrix=projection_matrix,
                                                              ref_pts=self.ref_pts.clone())  # B, H,W,C

        if self.merge_type == MergeType.concat:
            voxels = torch.cat((cross_wd.unsqueeze(1).repeat(1, self.vox_size[0], 1, 1, 1),
                                cross_hd.unsqueeze(2).repeat(1, 1, self.vox_size[1], 1, 1),
                                cross_hw.unsqueeze(3).repeat(1, 1, 1, self.vox_size[2], 1)), dim=-1) # B, H,W,D, C*3
        elif self.merge_type == MergeType.sum:
            voxels = cross_wd.unsqueeze(1) + cross_hd.unsqueeze(2) + cross_hw.unsqueeze(3)
        else:
            raise Exception("the merge type is not defined")
        voxels = self.merge(voxels) # B, H,W,D, C

        if self.residual and proposal is not None:
            voxels[0][indices[0], indices[1], indices[2], :] *= vox_feats[0]
            voxels = normalize(voxels, dim=-1)
            voxels = self.residual_net(voxels) # B, H,W,D, C
        # voxels = normalize(voxels, dim=-1)
        return voxels
