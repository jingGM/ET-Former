import numpy as np
from PIL import Image
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

from utilities.functions import to_device


class ConvolutionAlignment(nn.Module):
    def __init__(self, reference_ratio, in_dim, out_dim, heads_num, depth_num, kernel_size=7):
        super().__init__()
        self.heads_num = heads_num
        self.depth_num = depth_num
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim * heads_num, kernel_size=kernel_size,
                              stride=reference_ratio, padding=int((kernel_size - 1) / 2))
        self.linear = nn.Sequential(
            nn.Linear(out_dim, out_dim * depth_num), nn.LeakyReLU(),
            nn.Linear(out_dim * depth_num, out_dim * depth_num), nn.GELU(),
        )

        self.offset = nn.Sequential(nn.Linear(out_dim, 2), nn.Tanh())

    def forward(self, plane_fts):
        conv_out = self.conv(plane_fts)
        fts = rearrange(conv_out, "b (h c) l w -> (b h) l w c", h=self.heads_num).contiguous()
        linear_out = self.linear(fts)  # Bh, d, l, w, 2
        linear_out = rearrange(linear_out, "b l w (d c) -> b l w d c", d=self.depth_num).contiguous()
        offset = self.offset(linear_out)
        return offset


class MultiHeads(nn.Module):
    def __init__(self, heads_num, q_dim, v_dim, out_dim=None, proj_drop=0.0, attn_drop=0.0):
        super().__init__()
        self.heads_num = heads_num
        # self.out_dim = out_dim
        if out_dim is None:
            out_dim = q_dim
        self.out_dim = out_dim
        self.scale = out_dim ** -0.5
        self.proj_q_c = nn.Conv2d(q_dim, q_dim * heads_num, kernel_size=1, stride=1, padding=0)
        self.proj_q_l = nn.Sequential(nn.Linear(q_dim, q_dim), nn.LeakyReLU(), nn.Linear(q_dim, out_dim), nn.GELU())

        self.proj_k = nn.Sequential(nn.Linear(v_dim, v_dim), nn.LeakyReLU(),
                                    nn.Linear(v_dim, out_dim), nn.GELU())

        self.proj_v = nn.Sequential(nn.Linear(v_dim, v_dim), nn.LeakyReLU(),
                                    nn.Linear(v_dim, out_dim), nn.GELU())
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        self.proj_out = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.final_out = nn.Sequential(
            nn.Linear(self.out_dim * self.heads_num, self.out_dim), nn.LeakyReLU(),
            nn.Linear(self.out_dim, self.out_dim), nn.LeakyReLU(),
        )

    def forward(self, query, fts):
        B, C, H, W = query.shape
        query_1 = self.proj_q_c(query).contiguous()
        q = self.proj_q_l(rearrange(
            query_1, "b (h c) l w -> (b h) (l w) c", h=self.heads_num)).permute(0, 2, 1).contiguous()  # bh,C,M

        k = self.proj_k(fts.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # bh, C, N
        v = self.proj_v(fts.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # bh, C, N

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # Bh, qn, rn
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        enhanced = torch.einsum('b m n, b c n -> b c m', attn, v)
        enhanced = enhanced.reshape(B * self.heads_num, self.out_dim, H, W).contiguous()

        head_out = self.proj_drop(self.proj_out(enhanced))  # BH, C,H,W
        head_out = rearrange(head_out, "(b h) c l w -> b l w (h c)", h=self.heads_num)

        out = self.final_out(head_out)  # B,H,W,C
        return out


class CrossDeformableAttention(nn.Module):
    def __init__(self, vox_size, image_size, reference_num, device, reference_ratio, heads_num, num_levels,
                 in_dim, out_dim, kernel_size):
        super().__init__()
        self.vox_size = vox_size
        self.image_size = image_size
        self.device = device
        self.reference_num = reference_num
        self.num_levels = num_levels
        self.heads_num = heads_num
        self.align_hw = ConvolutionAlignment(reference_ratio=reference_ratio, in_dim=in_dim, out_dim=out_dim,
                                             kernel_size=kernel_size, heads_num=heads_num,
                                             depth_num=self.reference_num[2])
        self.align_hd = ConvolutionAlignment(reference_ratio=reference_ratio, in_dim=in_dim, out_dim=out_dim,
                                             kernel_size=kernel_size, heads_num=heads_num,
                                             depth_num=self.reference_num[1])
        self.align_wd = ConvolutionAlignment(reference_ratio=reference_ratio, in_dim=in_dim, out_dim=out_dim,
                                             kernel_size=kernel_size, heads_num=heads_num,
                                             depth_num=self.reference_num[0])

        self.multi_head_hw = MultiHeads(heads_num=heads_num, q_dim=in_dim, v_dim=in_dim, out_dim=out_dim)
        self.multi_head_hd = MultiHeads(heads_num=heads_num, q_dim=in_dim, v_dim=in_dim, out_dim=out_dim)
        self.multi_head_wd = MultiHeads(heads_num=heads_num, q_dim=in_dim, v_dim=in_dim, out_dim=out_dim)

    def image_mask_points(self, ref_pts, projection_matrix, pc_range):
        ref_pts[..., 0:1] = ref_pts[..., 0:1] * (pc_range[..., 3] - pc_range[..., 0]) + pc_range[..., 0]
        ref_pts[..., 1:2] = ref_pts[..., 1:2] * (pc_range[..., 4] - pc_range[..., 1]) + pc_range[..., 1]
        ref_pts[..., 2:3] = ref_pts[..., 2:3] * (pc_range[..., 5] - pc_range[..., 2]) + pc_range[..., 2]
        # ref_pts[..., 0:1] = ref_pts[..., 0:1] * (self.pc_range[..., 3] - self.pc_range[0]) + self.pc_range[0]
        # ref_pts[..., 1:2] = ref_pts[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        # ref_pts[..., 2:3] = ref_pts[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
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

        # image = torch.zeros((self.image_size[0], self.image_size[1], 3)).numpy()
        # reference_points_image[..., 0] *= self.image_size[1]
        # reference_points_image[..., 1] *= self.image_size[0]
        # ref_mask = (ref_mask & (reference_points_image[..., 1:2] > 0.0)
        #             & (reference_points_image[..., 1:2] < self.image_size[0])
        #             & (reference_points_image[..., 0:1] < self.image_size[1])
        #             & (reference_points_image[..., 0:1] > 0.0))
        # ref_mask = ref_mask.flatten()
        # reference_points_image = rearrange(reference_points_image, "x y z c -> (x y z) c")
        # pixels = reference_points_image[ref_mask].to(int).numpy()
        # image[pixels[:, 1], pixels[:, 0], 0] = 1
        # image[pixels[:, 1], pixels[:, 0], 1] = 1
        # image[pixels[:, 1], pixels[:, 0], 2] = 1
        # PIL_image = Image.fromarray(np.uint8(image * 255)).convert('RGB')
        # PIL_image.show()

        ref_mask = (ref_mask & (reference_points_image[..., 1:2] > 0.0)
                    & (reference_points_image[..., 1:2] < 1.0)
                    & (reference_points_image[..., 0:1] < 1.0)
                    & (reference_points_image[..., 0:1] > 0.0))
        ref_mask = ref_mask.flatten()
        reference_points_image = rearrange(reference_points_image, "x y z c -> (x y z) c")
        return reference_points_image[ref_mask][..., (1, 0)], ref_mask

    def get_offset_attention(self, hw_self_fts, hd_self_fts, wd_self_fts, ref_mask, reference_points_image):
        offset_hw = self.align_hw(hw_self_fts)  # Bh h w d c
        offset_hw = rearrange(offset_hw, "b x y z c -> b (x y z) c")[:, ref_mask, :]

        offset_hd = self.align_hd(hd_self_fts).permute(0, 1, 3, 2, 4).contiguous()  # Bh,h,d,w,c -> Bh, h, w, d, c
        offset_hd = rearrange(offset_hd, "b x y z c -> b (x y z) c")[:, ref_mask, :]

        offset_wd = self.align_wd(wd_self_fts).permute(0, 3, 1, 2, 4).contiguous()  # Bh,w,d,h,c -> Bh, h, w, d, c
        offset_wd = rearrange(offset_wd, "b x y z c -> b (x y z) c")[:, ref_mask, :]

        p_hw = (reference_points_image.unsqueeze(0) + offset_hw).clamp(0, +1.)  # b, xyz, 2
        p_hd = (reference_points_image.unsqueeze(0) + offset_hd).clamp(0, +1.)
        p_wd = (reference_points_image.unsqueeze(0) + offset_wd).clamp(0, +1.)
        return p_hw, p_hd, p_wd

    # def _apply_mask(self, reference_points_image, tpv_mask):

    def forward(self, hw_self_fts, hd_self_fts, wd_self_fts, img_feats, projection_matrix, ref_pts, pc_range):
        """
        args:
        hw_self_fts: B,C,H,W
        img_feats: B,C,X,Y
        ref_pts: Hr,Wr,Dr,3
        """
        reference_points_image, ref_mask = self.image_mask_points(ref_pts=ref_pts, projection_matrix=projection_matrix, pc_range=pc_range)
        reference_points_image = to_device(reference_points_image, self.device)
        p_hw, p_hd, p_wd = self.get_offset_attention(hw_self_fts=hw_self_fts, hd_self_fts=hd_self_fts,
                                                     wd_self_fts=wd_self_fts, ref_mask=ref_mask,
                                                     reference_points_image=reference_points_image)

        img_feats = rearrange(img_feats[0].repeat(1, self.heads_num, 1, 1, 1), "b n c x y -> (b n) c x y")
        sampled_hw = F.grid_sample(input=img_feats, grid=p_hw.unsqueeze(1), mode='bilinear',
                                   align_corners=True).squeeze(-2)  # bh, c, hwd
        sampled_hd = F.grid_sample(input=img_feats, grid=p_hd.unsqueeze(1), mode='bilinear',
                                   align_corners=True).squeeze(-2)  # bh, c, hwd
        sampled_wd = F.grid_sample(input=img_feats, grid=p_wd.unsqueeze(1), mode='bilinear',
                                   align_corners=True).squeeze(-2)  # bh, c, hwd

        cross_hw = self.multi_head_hw(query=hw_self_fts, fts=sampled_hw)
        cross_hd = self.multi_head_hw(query=hd_self_fts, fts=sampled_hd)
        cross_wd = self.multi_head_hw(query=wd_self_fts, fts=sampled_wd)
        return cross_hw, cross_hd, cross_wd
