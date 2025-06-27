from einops import rearrange
import numpy as np
import torch
from torch import nn
from torch.nn.functional import normalize
from mmcv.ops import knn
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16

from models.commons import BaseModel
from models.deformable_attn.pointconvformer import PCFLayer
from utilities.functions import to_device
from utilities.deformable_attn_configs import AggregationTypes


class LearnedPositionalEncoding(nn.Module):
    def __init__(self,
                 num_feats=128,
                 num_embed=(50, 50, 50), ):
        super(LearnedPositionalEncoding, self).__init__()
        self.h_embed = nn.Embedding(num_embed[0], num_feats)
        self.w_embed = nn.Embedding(num_embed[1], num_feats)
        self.d_embed = nn.Embedding(num_embed[2], num_feats)
        self.linear = nn.Sequential(nn.Linear(3 * num_feats, 2 * num_feats), nn.GELU(),
                                    nn.Linear(2 * num_feats, num_feats), nn.GELU(), nn.LayerNorm(num_feats))
        self.output = nn.Sequential(nn.Linear(3 * num_feats, 2 * num_feats), nn.GELU(),
                                    nn.Linear(2 * num_feats, num_feats), nn.GELU(), nn.LayerNorm(num_feats))
        # self.num_feats = num_feats
        # self.row_num_embed = row_num_embed
        # self.col_num_embed = col_num_embed

    def forward(self, poses):
        # b, n, c = poses.shape
        h = self.h_embed(poses[0])
        w = self.w_embed(poses[1])
        d = self.d_embed(poses[2])
        pos = torch.cat((h, w, d), dim=-1)
        comb = self.linear(pos)

        h_out = normalize(comb * h + h, dim=-1)
        w_out = normalize(comb * w + w, dim=-1)
        d_out = normalize(comb * d + d, dim=-1)
        output = self.output(torch.cat((h_out, w_out, d_out), dim=-1))
        return output


class MultiHeads(nn.Module):
    def __init__(self, heads_num, q_dim, v_dim, out_dim=None, proj_drop=0.0, attn_drop=0.0):
        super().__init__()
        self.heads_num = heads_num
        # self.out_dim = out_dim
        if out_dim is None:
            out_dim = q_dim
        self.out_dim = out_dim
        self.scale = out_dim ** -0.5
        self.proj_q = nn.Sequential(nn.Linear(q_dim, q_dim), nn.LeakyReLU(), nn.Linear(q_dim, out_dim), nn.GELU())

        self.proj_k = nn.Sequential(nn.Linear(v_dim, v_dim), nn.LeakyReLU(), nn.Linear(v_dim, out_dim), nn.GELU())

        self.proj_v = nn.Sequential(nn.Linear(v_dim, v_dim), nn.LeakyReLU(), nn.Linear(v_dim, out_dim), nn.GELU())
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        self.proj_out = nn.Sequential(nn.Linear(out_dim, out_dim), nn.LeakyReLU(), nn.Linear(out_dim, out_dim),
                                      nn.GELU())
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.final_out = nn.Sequential(
            nn.Linear(self.out_dim * self.heads_num, self.out_dim), nn.LeakyReLU(),
            nn.Linear(self.out_dim, self.out_dim), nn.LeakyReLU(),
        )

    def forward(self, query, fts):
        query = rearrange(self.proj_q(query).unsqueeze(1).repeat(1, self.heads_num, 1, 1),
                          "b h n c -> (b h) n c").permute(0, 2, 1)  # bh, C, N
        k = self.proj_k(fts.permute(0, 2, 1)).permute(0, 2, 1)  # bh, C, M
        v = self.proj_v(fts.permute(0, 2, 1)).permute(0, 2, 1)  # bh, C, M

        attn = torch.einsum('b c m, b c n -> b m n', query, k)  # Bh, N, M
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        enhanced = torch.einsum('b n m, b c m -> b c n', attn, v).permute(0, 2, 1)  # Bh, N, C

        head_out = self.proj_drop(self.proj_out(enhanced))  # BH, N, C
        head_out = rearrange(head_out, "(b h) n c -> b n (h c)", h=self.heads_num)

        out = self.final_out(head_out)  # B,N,C
        return out


class DeformableSparseAttn3D(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.voxel_size = config.vox_size
        self.reference_num = config.reference_num
        self.pc_range = config.pc_range
        self.image_size = config.image_size
        self.k_num = config.k_num
        self.reference_k_num = config.reference_k_num
        self.heads_num = config.heads_num
        self.aggregation_type = config.aggregation_type

        self.positional_embedding = LearnedPositionalEncoding(num_feats=config.pe_num_fts, num_embed=config.pe_num_emb)

        self.image_aggregation = nn.Conv2d(in_channels=config.pe_num_fts, kernel_size=config.patch_scale,
                                           padding="same", out_channels=config.pe_num_fts * self.heads_num)

        self.offset_network = PCFLayer(in_channel=config.pe_num_fts, out_channel=config.offset.out_dim,
                                       num_heads=config.offset.num_heads, batch_norm=config.offset.batch_norm,
                                       weightnet_output=config.offset.weightnet_output,
                                       output_dim=config.offset.offset_dim,
                                       guidance_feat_len=config.offset.guidance_feat_len, out_heads=self.heads_num,
                                       viewpoint_invariant=config.offset.viewpoint_invariant,
                                       layer_norm_guidance=config.offset.layer_norm_guidance)

        if self.aggregation_type == AggregationTypes.all:
            self.aggregation = MultiHeads(heads_num=self.heads_num, q_dim=config.pe_num_fts, v_dim=config.pe_num_fts,
                                          out_dim=config.pe_num_fts)
        elif self.aggregation_type == AggregationTypes.knn:
            self.aggregation = PCFLayer(in_channel=config.pe_num_fts, out_channel=config.pe_num_fts,
                                        num_heads=self.heads_num, batch_norm=False, activation=nn.GELU(),
                                        output_dim=config.pe_num_fts, out_heads=1,
                                        viewpoint_invariant=False, layer_norm_guidance=False)
            self.output = nn.Sequential(nn.Linear(config.pe_num_fts, config.pe_num_fts), nn.LeakyReLU(0.2),
                                        nn.Linear(config.pe_num_fts, config.pe_num_fts), nn.GELU())
        else:
            raise Exception("the aggregation type is not defined")

    @torch.no_grad()
    def _get_reference_points(self):
        x = torch.linspace(0.5, self.voxel_size[0] - 0.5, self.reference_num[0]).view(-1, 1, 1).expand(
            self.reference_num[0], self.reference_num[1], self.reference_num[2]) / self.voxel_size[0]
        y = torch.linspace(0.5, self.voxel_size[1] - 0.5, self.reference_num[1]).view(1, -1, 1).expand(
            self.reference_num[0], self.reference_num[1], self.reference_num[2]) / self.voxel_size[1]
        z = torch.linspace(0.5, self.voxel_size[2] - 0.5, self.reference_num[2]).view(1, 1, -1).expand(
            self.reference_num[0], self.reference_num[1], self.reference_num[2]) / self.voxel_size[2]
        ref_pts = torch.stack((x, y, z), -1)
        return ref_pts

    @torch.no_grad()
    def image_mask_points(self, ref_pts, projection_matrix):
        ref_pts[..., 0:1] = ref_pts[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        ref_pts[..., 1:2] = ref_pts[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        ref_pts[..., 2:3] = ref_pts[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        reference_points = torch.cat((ref_pts, torch.ones_like(ref_pts[..., :1])), -1)

        H, W, D, C = reference_points.shape
        reference_points_image = torch.matmul(projection_matrix.unsqueeze(0).repeat(H, W, D, 1, 1).to(torch.float32),
                                              reference_points.unsqueeze(-1).to(torch.float32)).squeeze(-1)

        # camera front
        eps = 1e-5
        ref_mask = (reference_points_image[..., 2:3] > eps)
        reference_points_image = reference_points_image[..., 0:2] / torch.maximum(
            reference_points_image[..., 2:3], torch.ones_like(reference_points_image[..., 2:3]) * eps)
        reference_points_image[..., 0] /= self.image_size[1]
        reference_points_image[..., 1] /= self.image_size[0]

        ref_mask = (ref_mask & (reference_points_image[..., 1:2] > 0.0)
                    & (reference_points_image[..., 1:2] < 1.0)
                    & (reference_points_image[..., 0:1] < 1.0)
                    & (reference_points_image[..., 0:1] > 0.0))
        ref_mask = ref_mask.flatten()
        reference_points_image = rearrange(reference_points_image, "x y z c -> (x y z) c")
        return reference_points_image[ref_mask][..., (1, 0)], ref_mask, ref_pts.view(-1, 3)[ref_mask]

    @torch.no_grad()
    def _process_points_relations(self, proposal, projection_matrix):
        ref_pts = self._get_reference_points()
        ref_image_pts, ref_mask, selected_ref_pts = self.image_mask_points(ref_pts=ref_pts,
                                                                           projection_matrix=projection_matrix)

        proposal = proposal.reshape(self.voxel_size[0], self.voxel_size[1], self.voxel_size[2])
        indices = np.asarray(np.where(proposal > 0))
        query_poses = torch.stack([
            torch.from_numpy(
                (indices[0] + 0.5) / self.voxel_size[0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]),
            torch.from_numpy(
                (indices[1] + 0.5) / self.voxel_size[1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]),
            torch.from_numpy(
                (indices[2] + 0.5) / self.voxel_size[2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
        ], dim=-1)

        return ref_image_pts, selected_ref_pts, query_poses, torch.from_numpy(indices)

    def forward(self, img_fts, proposal, projection_matrix):
        ref_image_pts, selected_ref_pts, query_poses, indices = self._process_points_relations(
            proposal=proposal, projection_matrix=projection_matrix)

        query_fts = self.positional_embedding(to_device(indices, device=self.device)).unsqueeze(0)  # N C
        # h_features = h * vox_feats
        # w_features = w * vox_feats
        # d_features = d * vox_feats

        ref_image_pts = to_device(ref_image_pts.unsqueeze(0).to(torch.float), device=self.device)
        query_poses = to_device(query_poses.unsqueeze(0).to(torch.float).contiguous(), device=self.device)
        selected_ref_pts = to_device(selected_ref_pts.unsqueeze(0).to(torch.float).contiguous(), device=self.device)
        knn_indices = knn(self.k_num, query_poses, selected_ref_pts)
        knn_indices = knn_indices.transpose(2, 1)
        offset = self.offset_network(dense_geos=query_poses, sparse_geos=selected_ref_pts,
                                     dense_feats=query_fts, nei_inds=knn_indices)
        new_ref_pts = (ref_image_pts + offset[0]).clamp(0, +1.)

        aggregated_im_fts = self.image_aggregation(img_fts[0][0])
        aggregated_im_fts = rearrange(aggregated_im_fts, "b (h c) x y -> (b h) c x y", h=self.heads_num)
        img_feats = rearrange(img_fts[0].repeat(1, self.heads_num, 1, 1, 1), "b n c x y -> (b n) c x y")
        # img_all_fts =   # F.normalize(aggregated_im_fts + img_feats, dim=1)
        sampled_fts = F.grid_sample(input=aggregated_im_fts + img_feats, grid=new_ref_pts.unsqueeze(1), mode='bilinear',
                                    align_corners=True).squeeze(-2)  # bh, c, hwd

        if self.aggregation_type == AggregationTypes.all:
            enhanced_q = self.aggregation(query=query_fts, fts=sampled_fts)
        elif self.aggregation_type == AggregationTypes.knn:
            ref_knn_indices = knn(self.reference_k_num, selected_ref_pts, query_poses).transpose(2, 1)
            aggregation = self.aggregation(dense_geos=selected_ref_pts, sparse_geos=query_poses,
                                           dense_feats=sampled_fts.transpose(2, 1), nei_inds=ref_knn_indices).squeeze(0)
            enhanced_q = self.output(aggregation + query_fts)
        else:
            raise Exception("the aggregation type is not defined")
        return enhanced_q
