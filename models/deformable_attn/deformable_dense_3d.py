import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from utilities.functions import to_device


class ConvolutionAlignment(nn.Module):
    def __init__(self, reference_ratio, in_dim, out_dim=32, heads_num=4, kernel_size=7):
        super().__init__()
        self.heads_num = heads_num
        self.conv = nn.Conv3d(in_channels=in_dim, out_channels=out_dim * heads_num, kernel_size=kernel_size,
                              stride=reference_ratio, padding=int((kernel_size - 1) / 2))

        self.offset = nn.Sequential(nn.Linear(out_dim, out_dim // 2), nn.GELU(),
                                    nn.Linear(out_dim // 2, 2), nn.Tanh())

    def forward(self, vox_fts):
        conv_out = self.conv(vox_fts)
        fts = rearrange(conv_out, "b (h c) l w d -> (b h) l w d c", h=self.heads_num)
        offset = self.offset(fts)
        return offset


class ConvolutionResidual(nn.Module):
    def __init__(self, reference_ratio, in_dim, out_dim=32, heads_num=4, kernel_size=7):
        super().__init__()
        self.heads_num = heads_num
        self.conv = nn.Conv3d(in_channels=in_dim, out_channels=out_dim * heads_num, kernel_size=kernel_size,
                              stride=reference_ratio, padding=int((kernel_size - 1) / 2))

        self.output = nn.Sequential(nn.Linear(out_dim, out_dim), nn.LeakyReLU(0.2),
                                    nn.Linear(out_dim, out_dim), nn.LeakyReLU())

    def forward(self, vox_fts):
        conv_out = self.conv(vox_fts)
        fts = rearrange(conv_out, "b (h c) l w d -> (b h) l w d c", h=self.heads_num)
        output = self.output(fts)
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

        self.proj_k = nn.Sequential(nn.Linear(v_dim, v_dim), nn.LeakyReLU(),
                                    nn.Linear(v_dim, out_dim // self.heads_num), nn.GELU())

        self.proj_v = nn.Sequential(nn.Linear(v_dim, v_dim), nn.LeakyReLU(),
                                    nn.Linear(v_dim, out_dim // self.heads_num), nn.GELU())
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        self.proj_out = nn.Sequential(nn.Linear(out_dim // self.heads_num, out_dim // self.heads_num), nn.LeakyReLU(),
                                      nn.Linear(out_dim // self.heads_num, out_dim // self.heads_num), nn.GELU())
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.final_out = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim), nn.LeakyReLU(),
            nn.Linear(self.out_dim, self.out_dim), nn.LeakyReLU(),
        )

    def forward(self, query, fts):
        query = rearrange(self.proj_q(query), "b n (h c) -> (b h) n c", h=self.heads_num).permute(0, 2, 1)  # bh, C, N
        k = self.proj_k(fts.permute(0, 2, 1)).permute(0, 2, 1)  # bh, C, M
        v = self.proj_v(fts.permute(0, 2, 1)).permute(0, 2, 1)  # bh, C, M

        attn = torch.einsum('b c m, b c n -> b m n', query, k)  # Bh, N, M
        attn *= self.scale
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        enhanced = torch.einsum('b n m, b c m -> b c n', attn, v).permute(0, 2, 1)  # Bh, N, C

        head_out = self.proj_drop(self.proj_out(enhanced))  # BH, N, C
        head_out = rearrange(head_out, "(b h) n c -> b n (h c)", h=self.heads_num)

        out = self.final_out(head_out)  # B,N,C
        return out


class DeformableDenseAttn3D(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config

        self.voxel_size = config.vox_size
        self.reference_num = config.reference_num
        self.pc_range = config.pc_range
        self.image_size = config.image_size
        self.heads_num = config.heads_num

        self.offset_network = ConvolutionAlignment(reference_ratio=config.off_ratio, in_dim=config.in_dim,
                                                   out_dim=config.off_out_dim, heads_num=config.off_head,
                                                   kernel_size=config.off_kernel)
        self.residual_network = ConvolutionResidual(reference_ratio=config.off_ratio, in_dim=config.in_dim,
                                                    out_dim=config.pe_num_fts, heads_num=config.off_head,
                                                    kernel_size=config.res_kernel)

        self.image_aggregation = nn.Conv2d(in_channels=config.pe_num_fts, kernel_size=config.patch_scale,
                                           padding="same", out_channels=config.pe_num_fts * config.off_head)

        self.multi_heads = MultiHeads(heads_num=config.off_head, q_dim=config.in_dim, v_dim=config.pe_num_fts,
                                      out_dim=config.mh_out_dim)

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
    def _get_reference_points(self):
        x = torch.linspace(0.5, self.voxel_size[0] - 0.5, self.reference_num[0]).view(-1, 1, 1).expand(
            self.reference_num[0], self.reference_num[1], self.reference_num[2]) / self.voxel_size[0]
        y = torch.linspace(0.5, self.voxel_size[1] - 0.5, self.reference_num[1]).view(1, -1, 1).expand(
            self.reference_num[0], self.reference_num[1], self.reference_num[2]) / self.voxel_size[1]
        z = torch.linspace(0.5, self.voxel_size[2] - 0.5, self.reference_num[2]).view(1, 1, -1).expand(
            self.reference_num[0], self.reference_num[1], self.reference_num[2]) / self.voxel_size[2]
        ref_pts = torch.stack((x, y, z), -1)
        return ref_pts

    def forward(self, vox_feats, img_fts, projection_matrix):
        offset = rearrange(self.offset_network(rearrange(vox_feats, "b x y z c -> b c x y z")),
                           "b l w d c -> b (l w d) c")
        residual = rearrange(self.residual_network(rearrange(vox_feats, "b x y z c -> b c x y z")),
                             "b l w d c -> b c (l w d)")

        ref_pts = self._get_reference_points()
        ref_image_pts, ref_mask, selected_ref_pts = self.image_mask_points(ref_pts=ref_pts,
                                                                           projection_matrix=projection_matrix)

        ref_image_pts = to_device(ref_image_pts.unsqueeze(0).to(torch.float), device=self.device)
        new_ref_pts = (ref_image_pts + offset[:, ref_mask, :]).clamp(0, +1.)

        aggregated_im_fts = self.image_aggregation(img_fts[0][0])
        aggregated_im_fts = rearrange(aggregated_im_fts, "b (h c) x y -> (b h) c x y", h=self.heads_num)
        img_feats = rearrange(img_fts[0].repeat(1, self.heads_num, 1, 1, 1), "b n c x y -> (b n) c x y")
        sampled_fts = F.grid_sample(input=aggregated_im_fts + img_feats, grid=new_ref_pts.unsqueeze(1), mode='bilinear',
                                    align_corners=True).squeeze(-2)  # bh, c, hwd
        sampled_fts += residual[:, :, ref_mask]
        output = self.multi_heads(query=rearrange(vox_feats, "b x y z c -> b (x y z) c"), fts=sampled_fts)
        return rearrange(output, "b (x y z) c -> b x y z c", x=vox_feats.shape[1], y=vox_feats.shape[2])
