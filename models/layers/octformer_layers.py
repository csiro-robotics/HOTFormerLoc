# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
#
# Adapted for use with carrier tokens by Ethan Griffiths.
# --------------------------------------------------------

import torch
from typing import Optional, List

import ocnn
from ocnn.octree import Octree
import dwconv

from models.octree import OctreeT
from models.layers.mask_powernorm import MaskPowerNorm



def get_norm_layer(channels: int, norm_type: str = 'batchnorm'):
    """
    Return the desired normalisation layer.
    """
    norm_type = norm_type.lower()
    if norm_type == 'batchnorm':
        norm_layer = torch.nn.BatchNorm1d(channels)
    elif norm_type == 'layernorm':
        norm_layer = torch.nn.LayerNorm(channels)
    elif norm_type == 'powernorm':
        norm_layer = MaskPowerNorm(channels)
    else:
        raise ValueError("Norm type must be either 'batchnorm' or 'layernorm'")
    return norm_layer


class MLP(torch.nn.Module):

    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, activation=torch.nn.GELU,
                 drop: float = 0.0, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.fc1 = torch.nn.Linear(self.in_features, self.hidden_features)
        self.act = activation()
        self.fc2 = torch.nn.Linear(self.hidden_features, self.out_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, data: torch.Tensor):
        data = self.fc1(data)
        data = self.act(data)
        data = self.drop(data)
        data = self.fc2(data)
        data = self.drop(data)
        return data


class OctreeDWConvNorm(torch.nn.Module):
    """
    Sequence of Octree DWConv, and BatchNorm/LayerNorm.
    """

    def __init__(self, in_channels: int, kernel_size: List[int] = [3],
                 nempty: bool = False, conv_norm: str = 'batchnorm'):
        super().__init__()
        self.conv = dwconv.OctreeDWConv(
            in_channels, kernel_size, nempty, use_bias=False)
        self.norm = get_norm_layer(in_channels, conv_norm)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.norm(out)
        return out


class OctreeConvNormRelu(torch.nn.Module):
    """
    Sequence of Octree Conv, BatchNorm/LayerNorm, and Relu.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: List[int] = [3], stride: int = 1,
                 nempty: bool = False, conv_norm: str = 'batchnorm'):
        super().__init__()
        self.conv = ocnn.nn.OctreeConv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.norm = get_norm_layer(out_channels, conv_norm)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.norm(out)
        out = self.relu(out)
        return out


class OctreeDeconvNormRelu(torch.nn.Module):
    """
    Sequence of Octree Deconv, BatchNorm/LayerNorm, and Relu.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: List[int] = [3], stride: int = 1,
                 nempty: bool = False, conv_norm: str = 'batchnorm'):
        super().__init__()
        self.deconv = ocnn.nn.OctreeDeconv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.norm = get_norm_layer(out_channels, conv_norm)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.deconv(data, octree, depth)
        out = self.norm(out)
        out = self.relu(out)
        return out


class CPE(torch.nn.Module):
    """
    Conditional position encoding (CPE). Also supports the slightly bulkier
    xCPE from PointTransformerV3.
    """
    def __init__(self, dim: int, nempty: bool = False,
                 conv_norm: str = 'layernorm', xcpe: bool = False):  
        super().__init__()      
        if not xcpe:
            self.conv = dwconv.OctreeDWConv(dim, nempty=nempty, use_bias=False)
            self.linear = torch.nn.Identity()
        else:
            self.conv = ocnn.nn.OctreeConv(dim, dim, nempty=nempty, use_bias=True)
            self.linear = torch.nn.Linear(dim, dim)
        self.norm = get_norm_layer(dim, conv_norm)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.linear(out)
        out = self.norm(out)
        return out

class RPE(torch.nn.Module):

    def __init__(self, patch_size: int, num_heads: int, dilation: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.pos_bnd = self.get_pos_bnd(patch_size)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3*self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def get_pos_bnd(self, patch_size: int):
        return int(0.8 * patch_size * self.dilation**0.5)

    def xyz2idx(self, xyz: torch.Tensor):
        mul = torch.arange(3, device=xyz.device) * self.rpe_num
        xyz = xyz.clamp(-self.pos_bnd, self.pos_bnd)
        idx = xyz + (self.pos_bnd + mul)
        return idx

    def forward(self, xyz):
        idx = self.xyz2idx(xyz)
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out

    def extra_repr(self) -> str:
        return 'num_heads={}, pos_bnd={}, dilation={}'.format(
                        self.num_heads, self.pos_bnd, self.dilation)  # noqa


class ADaPE(torch.nn.Module):
    """Absolute Distribution-aware Position Encoding (ADaPE)

    Small MLP network encodes carrier token centroid position (in x,y,z) and
    covariance of underlying points.

    Loosely inspired by SuperGlue Keypoint Encoder:
    https://arxiv.org/pdf/1911.11763.
    """
    
    def __init__(self, dim: int, activation: torch.nn.Module = torch.nn.GELU,
                 mode: str = 'cov'):
        """
        Args:
            dim (int): Feature dimension size.
            activation (torch.nn.Module): Activation between MLP linear layers.
                Defaults to GELU.
            mode (str): Mode determines whether position, variance, or
                covariance is used (cumulative aggregation of those three).
                Values must be one of ['pos','var','cov'].
        """
        super().__init__()
        assert mode in ['pos','var','cov'], "Invalid mode provided"
        # Num feats = 3 (x,y,z) + 6 (upper tri of cov matrix: σx, σxy, σxz, σy, σyz, σz)
        mode_num_feat_dict = {'pos': 3, 'var': 6, 'cov': 9}
        in_feat = mode_num_feat_dict[mode]
        self.mlp = MLP(in_feat, dim, dim, activation=activation, drop=0.0)
        # TODO: add layer/batchnorm after? try dropout?

    def forward(self, octree: OctreeT, depth: int):
        # Pass distribution stats of all windows through MLP to get PE
        window_stats = octree.window_stats[depth]
        out = self.mlp(window_stats)
        return out


class OctreeDropPath(torch.nn.Module):
    r'''Drop paths (Stochastic Depth) per sample when applied in main path of
    residual blocks, following the logic of :func:`timm.models.layers.DropPath`.

    Args:
        drop_prob (int): The probability of dropping paths.
        nempty (bool): Indicate whether the input data only contains features of
            the non-empty octree nodes or not.
        scale_by_keep (bool): Whether to scale the kept features proportionally.
        dilated_windows (bool): Whether dilation is being used.
        use_ct (bool): Whether carrier tokens are being used.
    '''

    def __init__(self, drop_prob: float = 0.0, nempty: bool = False,
                 scale_by_keep: bool = True, dilated_windows: bool = False,
                 use_ct: bool = False):
        super().__init__()

        self.drop_prob = drop_prob
        self.nempty = nempty
        self.scale_by_keep = scale_by_keep
        self.dilated_windows = dilated_windows
        self.use_ct = use_ct

    def forward(self, data: torch.Tensor, octree: OctreeT,
                depth: Optional[int] = None,
                batch_id: Optional[torch.Tensor] = None):
        r''''''

        if self.drop_prob <= 0.0 or not self.training:
            return data

        if depth is None:
            assert batch_id is not None, (
                "Batch idx must be provided for multi-scale tokens (see " \
                "calc_rt_attn_mask in RelayTokenTransformerBlock)"
            )
        batch_size = octree.batch_size
        ndim = data.ndim
        K = data.size(1)  # for ndim = 3, 2nd dim is the window dim
        assert ndim in (2, 3), "Invalid num dimensions in input"
        keep_prob = 1 - self.drop_prob
        rnd_tensor = torch.rand(
            batch_size, 1, dtype=data.dtype, device=data.device
        )
        rnd_tensor = torch.floor(rnd_tensor + keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            rnd_tensor.div_(keep_prob)

        if batch_id is None:
            batch_id = octree.batch_id(depth, self.nempty)
            # Check if dealing with Octree batch or windowed/ct batch
            if self.use_ct:
                # Get batch id of each ct
                batch_id = octree.rt_batch_idx[depth]
            elif ndim == 3:
                if K == octree.patch_size:  # standard window attn
                    batch_id = octree.data_to_windows(
                        batch_id.unsqueeze(-1), depth, self.dilated_windows,
                        fill_value=(batch_size - 1)
                    ).squeeze(-1)
                    # NOTE: Padding is almost guaranteed to belong only to the
                    #       final batch elem (as long as num_windows >=
                    #       dilation), and it doesn't matter anyways since it is
                    #       just padding that will be getting dropped.
                else:  # HAT attn (window + RT)
                    batch_id = octree.hat_batch_window_idx[depth]
            # Assume padding idx as part of last batch
            batch_id = batch_id.minimum(torch.tensor(batch_size - 1))                    

        drop_mask = rnd_tensor[batch_id]
        output = data * drop_mask
        return output

    def extra_repr(self) -> str:
        return ('drop_prob={:.4f}, nempty={}, scale_by_keep={}').format(
                self.drop_prob, self.nempty, self.scale_by_keep)  # noqa
