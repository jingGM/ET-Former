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

import numpy as np
import torch
from torch import nn
import copy
import warnings
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader

from models.voxformer.feedforward import FFN
from models.voxformer.utils import build_norm_layer
from utilities.voxformer_configs import TransformerType
from models.voxformer.deformable_self_attention import DeformSelfAttention
from models.voxformer.deformable_cross_attention import DeformCrossAttention

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


class VoxFormerLayer(nn.Module):
    """Implements encoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 ffn_cfgs,
                 former_type,
                 batch_first=True,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ):
        # super(VoxFormerLayer, self).__init__(
        #     attn_cfgs=attn_cfgs,
        #     feedforward_channels=feedforward_channels,
        #     ffn_dropout=ffn_dropout,
        #     operation_order=operation_order,
        #     act_cfg=act_cfg,
        #     norm_cfg=norm_cfg,
        #     ffn_num_fcs=ffn_num_fcs,
        #     **kwargs)
        super(VoxFormerLayer, self).__init__()
        self.fp16_enabled = False
        # deprecated_args = dict(
        #     feedforward_channels='feedforward_channels',
        #     ffn_dropout='ffn_drop',
        #     ffn_num_fcs='num_fcs')
        # for ori_name, new_name in deprecated_args.items():
        #     if ori_name in kwargs:
        #         warnings.warn(
        #             f'The arguments `{ori_name}` in BaseTransformerLayer '
        #             f'has been deprecated, now you should set `{new_name}` '
        #             f'and other FFN related arguments '
        #             f'to a dict named `ffn_cfgs`. ')
        #         ffn_cfgs[new_name] = kwargs[ori_name]

        # super(MyCustomBaseTransformerLayer, self).__init__(init_cfg)

        self.batch_first = batch_first
        num_attn = operation_order.count('self_attn') + operation_order.count('cross_attn')
        self._check_availability(num_attn, operation_order, attn_cfgs)

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'

        self.attentions = nn.ModuleList()
        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                if former_type == TransformerType.self:
                    attention = DeformSelfAttention(embed_dims=attn_cfgs[index].embed_dims, num_heads=8,
                                                    num_levels=attn_cfgs[index].num_levels,
                                                    num_points=attn_cfgs[index].num_points,
                                                    num_bev_queue=2, im2col_step=64, dropout=0.1, batch_first=True,
                                                    norm_cfg=None, init_cfg=None)
                elif former_type == TransformerType.cross:
                    attention = DeformCrossAttention(embed_dims=attn_cfgs[index].embed_dims,
                                                     num_cams=attn_cfgs[index].num_cams,
                                                     dropout=0.1, init_cfg=None, batch_first=False,
                                                     deformable_attention=attn_cfgs[index].deformable_attention)
                else:
                    raise Exception("the attention model is not defined")
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count('ffn')
        for ffn_index in range(num_ffns):
            self.ffns.append(FFN(embed_dims=ffn_cfgs.embed_dims, feedforward_channels=ffn_cfgs.feedforward_channels,
                                 num_fcs=ffn_cfgs.num_fcs, act_cfg=ffn_cfgs.act_cfg, ffn_drop=ffn_cfgs.ffn_drop))

        self.norms = nn.ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def _check_availability(self, num_attn, operation_order, attn_cfgs):
        assert set(operation_order) & set(
            ['self_attn', 'norm', 'ffn', 'cross_attn']) == \
               set(operation_order), f'The operation_order of' \
                                     f' {self.__class__.__name__} should ' \
                                     f'contains all four operation type ' \
                                     f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                                               f'of attn_cfg {num_attn} is ' \
                                               f'not consistent with the number of attention' \
                                               f'in operation_order {operation_order}.'

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerEncoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


class VoxFormerEncoder(nn.Module):
    """
    Attention with both self and cross
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, transformerlayers, num_layers,
                 image_size, num_points_in_pillar=4, return_intermediate=False):

        super(VoxFormerEncoder, self).__init__()
        self.return_intermediate = return_intermediate
        self.image_size = image_size

        self.num_points_in_pillar = num_points_in_pillar
        self.fp16_enabled = False
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(VoxFormerLayer(attn_cfgs=transformerlayers.attn_cfgs, ffn_cfgs=transformerlayers.ffn_cfgs,
                                              former_type=transformerlayers.type, batch_first=True,
                                              operation_order=transformerlayers.operation_order,
                                              norm_cfg=dict(type='LN')
                                              )
                               )
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in DCA and DSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                                          torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device))
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range, lidar2img):
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()
        pc_range = torch.from_numpy(pc_range).unsqueeze(0).unsqueeze(0)
        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[..., 3] - pc_range[..., 0]) + pc_range[..., 0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[..., 4] - pc_range[..., 1]) + pc_range[..., 1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[..., 5] - pc_range[..., 2]) + pc_range[..., 2]

        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= self.image_size[1]
        reference_points_cam[..., 1] /= self.image_size[0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                lidar2img,
                pc_range,
                *args,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerEncoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        ref_2d = self.get_reference_points(512, 512, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        bs, len_bev, num_bev_level, _ = ref_2d.shape

        hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs * 2, len_bev, num_bev_level, 2)

        reference_points_cam, bev_mask = self.point_sampling(ref_3d, pc_range, lidar2img)

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam.to(bev_query.device),
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
