import math

import torch
from torch import nn
from torch.nn.functional import pad

from utilities.functions import to_2tuple
from models.triplane.utils import DropPath, LayerScale, trunc_normal_
from natten.functional import NATTEN2DQKRPBFunction, NATTEN2DAVFunction
from models.triplane.dat_blocks import LayerNormProxy, TransformerMLP, TransformerMLPWithConv, LocalAttention, DAttentionBaseline, ShiftWindowAttention, PyramidAttention


class NeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """
    def __init__(self, dim, kernel_size, num_heads, attn_drop=0., proj_drop=0.,
                 dilation=None):
        super().__init__()
        self.fp16_enabled = False
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size
        if type(dilation) is str:
            self.dilation = None
            self.window_size = None
        else:
            assert dilation is None or dilation >= 1, \
                f"Dilation must be greater than or equal to 1, got {dilation}."
            self.dilation = dilation or 1
            self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Linear(dim, dim * 3)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # assert x.dtype == torch.float16, f"AMP failed!, dtype={x.dtype}"
        x = x.permute(0, 2, 3, 1)
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        dilation = self.dilation
        window_size = self.window_size
        if window_size is None:
            dilation = max(min(H, W) // self.kernel_size, 1)
            window_size = dilation * self.kernel_size
        if H < window_size or W < window_size:
            pad_l = pad_t = 0
            pad_r = max(0, window_size - W)
            pad_b = max(0, window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        # breakpoint()
        attn = NATTEN2DQKRPBFunction.apply(q, k, self.rpb, self.kernel_size, dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = NATTEN2DAVFunction.apply(attn, v, self.kernel_size, dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x)).permute(0, 3, 1, 2), None, None


class SelfDeformableAttention(nn.Module):
    def __init__(self, fmap_size, window_size,
                 dim_in, dim_embed, depths, stage_spec, n_groups,
                 use_pe, sr_ratio,
                 heads, stride,
                 offset_range_factor,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate,
                 use_dwc_mlp, ksize, nat_ksize,
                 layer_scale_value, use_lpu, log_cpb):

        super().__init__()
        # fmap_size = to_2tuple(fmap_size)
        assert len(fmap_size) == len(window_size) == 2, "window size and fmap size should all be 2"
        smallest_window_size = min(window_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()
        self.stage_spec = stage_spec
        self.use_lpu = use_lpu

        self.ln_cnvnxt = nn.ModuleDict(
            {str(d): LayerNormProxy(dim_embed) for d in range(depths) if stage_spec[d] == 'X'}
        )
        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) if stage_spec[d // 2] != 'X' else nn.Identity() for d in range(2 * depths)]
        )

        mlp_fn = TransformerMLPWithConv if use_dwc_mlp else TransformerMLP

        self.mlps = nn.ModuleList(
            [
                mlp_fn(dim_embed, expansion, drop) for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        self.layer_scales = nn.ModuleList(
            [
                LayerScale(dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity()
                for _ in range(2 * depths)
            ]
        )
        self.local_perception_units = nn.ModuleList(
            [
                nn.Conv2d(dim_embed, dim_embed, kernel_size=3, stride=1, padding=1,
                          groups=dim_embed) if use_lpu else nn.Identity()
                for _ in range(depths)
            ]
        )

        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads,
                                       hc, n_groups, attn_drop, proj_drop,
                                       stride, offset_range_factor, use_pe, dwc_pe,
                                       no_off, fixed_pe, ksize, log_cpb)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(smallest_window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            elif stage_spec[i] == 'N':
                self.attns.append(
                    NeighborhoodAttention2D(dim_embed, nat_ksize, heads, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'P':
                self.attns.append(
                    PyramidAttention(dim_embed, heads, attn_drop, proj_drop, sr_ratio)
                )
            elif self.stage_spec[i] == 'X':
                self.attns.append(
                    nn.Conv2d(dim_embed, dim_embed, kernel_size=smallest_window_size,
                              padding=smallest_window_size // 2, groups=dim_embed)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')

            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def forward(self, x):

        x = self.proj(x)

        for d in range(self.depths):

            if self.use_lpu:
                x0 = x
                x = self.local_perception_units[d](x.contiguous())
                x = x + x0

            if self.stage_spec[d] == 'X':
                x0 = x
                x = self.attns[d](x)
                x = self.mlps[d](self.ln_cnvnxt[str(d)](x))
                x = self.drop_path[d](x) + x0
            else:
                x0 = x
                x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
                x = self.layer_scales[2 * d](x)
                x = self.drop_path[d](x) + x0
                x0 = x
                x = self.mlps[d](self.layer_norms[2 * d + 1](x))
                x = self.layer_scales[2 * d + 1](x)
                x = self.drop_path[d](x) + x0

        return x