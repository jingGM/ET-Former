
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def index_batched_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx.to(torch.int64)[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class BatchNormKNN(nn.Module):
    def __init__(self, in_channel, out_channel, momentum=0.1):
        super(BatchNormKNN, self).__init__()
        self.linear = nn.Linear(in_features=in_channel, out_features=out_channel)  # nn.Sequential(, nn.ReLU())
        self.batch_norm = nn.BatchNorm2d(num_features=out_channel, momentum=momentum)

    def forward(self, points):
        """
        Args:
            points: BxNxKxC
            lengths: list
        Returns:
        """
        points = self.linear(points)
        batch_pts = torch.permute(points, (0, 3, 1, 2))  # B,C,M,K
        after_norm = self.batch_norm(batch_pts)
        return torch.permute(after_norm, (0, 2, 3, 1))  # B,M,K,C


class BatchNormLinear(nn.Module):
    def __init__(self, in_channel, out_channel, momentum=0.1):
        super(BatchNormLinear, self).__init__()
        self.linear = nn.Linear(in_features=in_channel, out_features=out_channel)  # nn.Sequential(, nn.ReLU())
        self.batch_norm = nn.BatchNorm1d(num_features=out_channel, momentum=momentum)
        self.out_channel = out_channel

    def forward(self, points):
        """
        Args:
            points: BxNxC
            lengths: list
        Returns:
        """
        B, N, C = points.shape
        input_points = points.contiguous().view(-1, C)
        points = self.linear(input_points)
        after_norm = self.batch_norm(points)
        return after_norm.view(B, N, self.out_channel)


# Multi-head Guidance:
# Input: guidance_query: input features (B x N x K x C)
#        guidance_key: also input features (but less features when downsampling)
#        pos_encoding: if not None, then position encoding is concatenated with the features
# Output: guidance_features: (B x N x K x num_heads)
class MultiHeadGuidance(nn.Module):
    """ Multi-head guidance to increase model expressivitiy"""

    def __init__(self, batch_norm: bool, layer_norm_guidance: bool, num_heads: int, num_hiddens: int):
        super(MultiHeadGuidance, self).__init__()
        self.dim = num_hiddens
        self.num_heads = num_heads
        self.batch_norm = batch_norm

        self.layer_norm_q = nn.LayerNorm(num_hiddens) if layer_norm_guidance else nn.Identity()
        self.layer_norm_k = nn.LayerNorm(num_hiddens) if layer_norm_guidance else nn.Identity()

        self.mlp = nn.ModuleList()
        mlp_dim = [self.dim, 8, num_heads]
        for ch_in, ch_out in zip(mlp_dim[:-1], mlp_dim[1:]):
            if batch_norm:
                self.mlp.append(BatchNormKNN(ch_in, ch_out))
            else:
                self.mlp.append(nn.Linear(ch_in, ch_out))

    def forward(self, guidance_query, guidance_key):  # , pos_encoding=None):
        # attention nxkxc
        scores = self.layer_norm_q(guidance_query) - self.layer_norm_k(guidance_key)
        for i, layer in enumerate(self.mlp):
            if self.batch_norm:
                scores = layer(scores)
            else:
                scores = layer(scores)
            if i == len(self.mlp) - 1:
                scores = torch.sigmoid(scores)
            else:
                scores = F.relu(scores, inplace=True)
        return scores


# Multi-head Guidance using the inner product of QK, as in conventional attention models. However,
# a sigmoid function is used as activation
# Input: guidance_query: input features (B x N x K x C)
#        guidance_key: also input features (but less features when downsampling)
#        pos_encoding: if not None, then position encoding is concatenated with the features
# Output: guidance_features: (B x N x K x num_heads)
class MultiHeadGuidanceQK(nn.Module):
    """ Multi-head guidance to increase model expressivitiy"""

    def __init__(self, num_heads: int, num_hiddens: int, key_dim: int):
        super(MultiHeadGuidanceQK, self).__init__()
        assert num_hiddens % num_heads == 0, 'num_hiddens: %d, num_heads: %d' % (
            num_hiddens, num_heads)
        self.dim = num_hiddens
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale = self.key_dim ** -0.5
        self.qk_linear = BatchNormKNN(self.dim, key_dim * num_heads)

    def forward(self, q, k):
        # compute q, k
        B, N, K, _ = q.shape

        q = self.qk_linear(q)
        k = self.qk_linear(k)
        q = q.view(B, N, K, self.num_heads, -1)
        k = k.view(B, N, K, self.num_heads, -1)
        # actually there is only one center..
        k = k[:, :, :1, :, :]
        q = q.transpose(2, 3)  # B,N,H,Kq,D
        k = k.permute(0, 1, 3, 4, 2)  # B,N,H,D, Kk

        # compute attention
        attn_score = (q @ k) * self.scale
        attn_score = attn_score[:, :, :, :, 0].transpose(2, 3)
        # Disabled softmax version since it performs significantly worse
        attn_score = torch.sigmoid(attn_score)
        return attn_score


class WeightNet(nn.Module):
    '''
    WeightNet for PointConv. This runs a few MLP layers (defined by hidden_unit) on the
    point coordinates and outputs generated weights for each neighbor of each point.
    The weights will then be matrix-multiplied with the input to perform convolution

    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        hidden_unit: Number of hidden units, a list which can contain multiple hidden layers
        efficient: If set to True, then gradient checkpointing is used in training to reduce memory cost
    Input: Coordinates for all the kNN neighborhoods
           input shape is B x N x K x in_channel, B is batch size, in_channel is the dimensionality of
            the coordinates (usually 3 for 3D or 2 for 2D, 12 for VI), K is the neighborhood size,
            N is the number of points
    Output: The generated weights B x N x K x C_mid
    '''

    def __init__(self, in_channel, out_channel, hidden_unit=[9, 16], efficient=False):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.efficient = efficient
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(BatchNormKNN(in_channel, out_channel))
        else:
            hidden_unit = [in_channel] + list(hidden_unit) + [out_channel]
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(BatchNormKNN(hidden_unit[i - 1], hidden_unit[i]))

    def forward(self, localized_xyz):
        """
        Args:
            localized_xyz: BxNxKxC
        """
        for conv in self.mlp_convs:
            localized_xyz = conv(localized_xyz)
            localized_xyz = F.relu(localized_xyz, inplace=True)
        return localized_xyz


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn, no_relu=False, momentum=0.1):
        """
        Initialize a standard unary block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """
        super(LinearBlock, self).__init__()
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_bn = use_bn
        if use_bn:
            self.mlp = BatchNormLinear(in_dim, out_dim, momentum=momentum)
        else:
            self.mlp = nn.Linear(in_dim, out_dim)
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        else:
            self.leaky_relu = nn.Identity()
        return

    def forward(self, x):
        x = self.mlp(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x


class PCFLayer(nn.Module):
    '''
    PointConvFormer main layer
    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        weightnet: Number of input/output channels for weightnet
        num_heads: Number of heads
        guidance_feat_len: Number of dimensions of the query/key features
    Input:
        dense_xyz: tensor (batch_size, num_points, 3). The coordinates of the points before subsampling
                   (if it is a "strided" convolution wihch simultaneously subsamples the point cloud)
        dense_feats: tensor (batch_size, num_points, num_dims). The features of the points before subsampling.
        nei_inds: tensor (batch_size, num_points2, K). The neighborhood indices of the K nearest neighbors
                  of each point (after subsampling). The indices should index into dense_xyz and dense_feats,
                  as during subsampling features at new coordinates are aggregated from the points before subsampling
        dense_xyz_norm: tensor (batch_size, num_points, 3). The surface normals of the points before subsampling
        sparse_xyz: tensor (batch_size, num_points2, 3). The coordinates of the points after subsampling (if there
                    is no subsampling, just input None for this and the next)
        sparse_xyz_norm: tensor (batch_size, num_points2, 3). The surface normals of the points after subsampling
        vi_features: tensor (batch_size, num_points2, 12). VI features only needs to be computed once per stage. If
                     it has been computed in a previous layer, it can be saved and directly inputted here.
        Note: batch_size is usually 1 since we are using the packed representation packing multiple point clouds into one. However this dimension needs to be there for pyTorch to work properly.
    Output:
        new_feat: output features
        weightNetInput: the input to weightNet, which are relative coordinates or viewpoint-invariance aware transforms of it
    '''

    def __init__(self, in_channel, out_channel, weightnet_output=16, num_heads=4, guidance_feat_len=32,
                 batch_norm=True, viewpoint_invariant=False, output_dim=2, out_heads=4, activation=nn.Tanh(),
                 layer_norm_guidance=False):
        super(PCFLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_heads = num_heads
        self.out_heads = out_heads
        self.viewpoint_invariant = viewpoint_invariant

        # First downscaling mlp
        self.voxel_fts_0 = LinearBlock(in_channel, out_channel // 4, use_bn=True, momentum=0.1)
        self.voxel_fts_1 = LinearBlock(out_channel // 4, guidance_feat_len, use_bn=True, momentum=0.1, no_relu=True)

        self.batch_norm = batch_norm
        if self.viewpoint_invariant:
            weight_input_channel = 12
        else:
            weight_input_channel = 3
        if self.batch_norm:
            self.weight_fts = BatchNormKNN(weight_input_channel, guidance_feat_len)
        else:
            self.weight_fts = nn.Linear(weight_input_channel, guidance_feat_len)

        # check last_ch % num_heads == 0
        assert (out_channel // 2) % num_heads == 0, "the out channel {} // 2 % heads{} should be 0".format(out_channel,
                                                                                                           num_heads)
        # if attention_type == AttentionType.subtraction:
        self.guidance_weight = MultiHeadGuidance(batch_norm=batch_norm, layer_norm_guidance=layer_norm_guidance,
                                                 num_heads=num_heads, num_hiddens=2 * guidance_feat_len)

        self.weightnet = WeightNet(weight_input_channel, weightnet_output)

        if self.batch_norm:
            self.output_1 = BatchNormLinear(out_channel // 4 * weightnet_output, out_channel // 2)
        else:
            self.output_1 = nn.Linear(out_channel // 4 * weightnet_output, out_channel // 2)

        # if self.out_heads > 1:
        self.output_2 = LinearBlock(out_channel // 2, out_channel, use_bn=True, momentum=0.1, no_relu=True)

        self.linear_shortcut = LinearBlock(in_channel, out_channel, use_bn=True, momentum=0.1, no_relu=True)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.offset = nn.Sequential(nn.Linear(out_channel // out_heads, output_dim), activation)
        # else:
        #     self.output_2 = LinearBlock(out_channel // 2, out_channel // self.num_heads, use_bn=True, momentum=0.1, no_relu=True)
        #     self.linear_shortcut = LinearBlock(in_channel, out_channel, use_bn=True, momentum=0.1, no_relu=True)
        #     self.leaky_relu = nn.LeakyReLU(0.1)
        #     self.offset = nn.Sequential(nn.Linear(out_channel, output_dim), activation)

    def forward(self, dense_geos, sparse_geos, dense_feats, nei_inds):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        dense_xyz_norm: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        sparse_xyz_norm: tensor (batch_size, num_points2, 3)
        """

        dense_xyz = dense_geos[:, :, :3]
        sparse_xyz = sparse_geos[:, :, :3]

        B, N, C_d = dense_xyz.shape
        B, M, C_s = sparse_xyz.shape
        B, M_k, K = nei_inds.shape
        assert (M == M_k) and (C_d == C_s == 3), "the number of the desnse and sparse points are not the same as KNN"

        gathered_xyz = index_batched_points(points=dense_xyz, idx=nei_inds)  # [B, M, K, 3]
        localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=-2)  # [B, M, K, 3]
        # Encode weightNetInput to be higher dimensional to match with gathered feat
        feat_pe = self.weight_fts(localized_xyz)
        feat_pe = F.relu(feat_pe)  # B,M,K,guidance_feat_len

        # first downscaling mlp
        feats_x = self.voxel_fts_0(dense_feats)  # B,N, out_channel // 4
        guidance_x = self.voxel_fts_1(feats_x)  # B, N, guidance_feat_len
        # Gather features on this low dimensionality is faster and uses less memory
        gathered_feat2 = index_batched_points(guidance_x, nei_inds)  # [B,M, K, guidance_feat_len]
        guidance_feature = torch.cat([gathered_feat2, feat_pe], dim=-1)  # [B,M, K, 2*guidance_feat_len]

        guidance_query = guidance_feature  # b m k 2*guidance_feat_len
        if M == N:
            guidance_key = guidance_feature[:, :, :1, :].repeat(1, 1, K, 1)  # b m k 2*guidance_feat_len
        else:
            guidance_key = guidance_feature.max(dim=2, keepdim=True)[0].repeat(1, 1, K, 1)  # b m k 2*guidance_feat_len

        guidance_score = self.guidance_weight(guidance_query, guidance_key)  # B M k num_heads
        # WeightNet computes the convolutional weights
        weights = self.weightnet(localized_xyz)  # B,M,K,weightnet_output

        gathered_feat = index_batched_points(feats_x, nei_inds)  # [B, M, K, out_channel // 4]
        gathered_feat = gathered_feat.permute(0, 3, 2, 1)  # B,C,K,M
        gathered_feat = (gathered_feat.view(B, -1, self.num_heads, K, M) * guidance_score.permute(0, 3, 2, 1).unsqueeze(1))
        gathered_feat = gathered_feat.view(B, -1, K, M).permute(0, 3, 1, 2).contiguous()  # B,C,K,M -> B,M,C,K
        # B,M,C,K x B,M,K,C -> B,M,C,C -> B,M,C**2
        gathered_feat = torch.matmul(input=gathered_feat, other=weights).view(B, M, -1)

        new_feat = F.relu(self.output_1(gathered_feat), inplace=True)  # B,M,C

    # if self.out_heads > 1:
        new_feat = self.output_2(new_feat)  # B,M,C
        new_feat = rearrange(new_feat, "b n (c h) -> b h n c", h=self.out_heads)

        sparse_feats = torch.max(index_batched_points(dense_feats, nei_inds), dim=2)[0]  # B,M,1,C
        shortcut = self.linear_shortcut(sparse_feats)  # B,M,C
        shortcut = rearrange(shortcut, "b n (c h) -> b h n c", h=self.out_heads)
        new_feat = self.leaky_relu(new_feat + shortcut)
        offset = self.offset(new_feat)
        # else:
        #     new_feat = self.output_2(new_feat)  # B,M,C
        #
        #     sparse_feats = torch.max(index_batched_points(dense_feats, nei_inds), dim=2)[0]  # B,M,1,C
        #     shortcut = self.linear_shortcut(sparse_feats)  # B,M,C
        #     # shortcut = rearrange(shortcut, "b n (c h) -> b h n c", h=self.out_heads)
        #     new_feat = self.leaky_relu(new_feat + shortcut)
        #     offset = self.offset(new_feat)
        return offset
