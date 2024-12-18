# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
#
# Adapted from https://github.com/octree-nn/octformer by
# Ethan Griffiths (Data61, Pullenvale)
# --------------------------------------------------------

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import unpad_sequence
import ocnn

from ocnn.octree import Octree
from typing import Optional, List
from torch.utils.checkpoint import checkpoint
from models.octree import OctreeT, pad_sequence
from models.layers.octformer_layers import get_norm_layer, MLP, \
    OctreeConvNormRelu, OctreeDeconvNormRelu, CPE, RPE, ADaPE, OctreeDropPath


class OctreeAttention(torch.nn.Module):

    def __init__(self, dim: int, patch_size: int, num_heads: int,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 dilation: int = 1, rt_per_window: int = 0, use_rpe: bool = True):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.rt_per_window = rt_per_window
        self.use_rpe = use_rpe
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)

        # NOTE: self.rpe is not used in the original experiments of my paper. When
        # releasing the code, I added self.rpe because I observed that it could
        # stablize the training process and improve the performance on ScanNet by
        # 0.3 to 0.5; on the other datasets, the improvements are more marginal. So
        # it is not indispensible, and can be removed by setting `use_rpe` as False.
        self.rpe = RPE(patch_size, num_heads, dilation) if use_rpe else None

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        H = self.num_heads
        K = self.patch_size
        C = self.dim
        D = self.dilation
        G = self.rt_per_window

        if D > 1:  # dilation
            rel_pos = octree.dilate_pos[depth]
            mask = octree.dilate_mask[depth]
        else:
            rel_pos = octree.rel_pos[depth]
            if G > 0:  # get correct mask for HAT attention
                mask = octree.hat_window_mask[depth]
            else:
                mask = octree.patch_mask[depth]

        # qkv
        qkv = self.qkv(data).reshape(-1, K+G, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]      # (N, H, K+G, C')
        q = q * self.scale

        # attn
        attn = q @ k.transpose(-2, -1)        # (N, H, K+G, K+G)
        attn = self.apply_rpe(attn, rel_pos)  # (N, H, K+G, K+G)
        attn = attn + mask.unsqueeze(1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        data = (attn @ v).transpose(1, 2).reshape(-1, K+G, C)  # (N, K+G, C)

        # ffn
        data = self.proj(data)
        data = self.proj_drop(data)
        return data

    def apply_rpe(self, attn, rel_pos):
        if self.use_rpe:
            rpe = self.rpe(rel_pos)
            if self.rt_per_window > 0:
                # Pad RPE for RTs (assume no relative pos for RTs)
                rpe = F.pad(rpe, (self.rt_per_window, 0, self.rt_per_window, 0))
            attn = attn + rpe
        return attn

    def extra_repr(self) -> str:
        return 'dim={}, patch_size={}, num_heads={}, dilation={}'.format(
                        self.dim, self.patch_size, self.num_heads, self.dilation)  # noqa


class RTAttentionOld(torch.nn.Module):
    def __init__(self, dim: int, patch_size: int, num_heads: int,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 rt_per_window: int = 0,
                 use_rpe: bool = True):
        """
        Deprecated version of relay token attention.
        """
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.rt_per_window = rt_per_window
        self.num_heads = num_heads
        self.use_rpe = use_rpe  # TODO: implement RPE
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)

        # NOTE: RPE table is currently constructed using relative pos of
        #       octree nodes, but this is not so easy to do for CTs. Need to
        #       find another solution.
        # self.rpe = RPE(patch_size, num_heads) if use_rpe else None

    def forward(self, relay_tokens: torch.Tensor, octree: OctreeT, depth: int):
        B = octree.batch_size
        H = self.num_heads
        C = self.dim
        rt = relay_tokens

        # split CTs into batches for each batch elem, padded to size of largest batch
        batch_num_windows = octree.batch_num_windows[depth]
        rt = rt.split(batch_num_windows.tolist())
        rt = pad_sequence(rt)

        # get RT attn mask
        rt_mask = octree.rt_mask[depth]

        # qkv
        qkv = self.qkv(rt).reshape(B, -1, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]      # (B, H, K, C')
        q = q * self.scale

        # attn
        attn = q @ k.transpose(-2, -1)        # (B, H, K, K)
        # attn = self.apply_rpe(attn, rel_pos)  # TODO: implement RPE
        attn = attn + rt_mask.unsqueeze(1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        rt = (attn @ v).transpose(1, 2).reshape(B, -1, C)  # (B, K, C)

        # Undo padding
        rt = torch.cat(unpad_sequence(rt, batch_num_windows, batch_first=True))

        # ffn
        rt = self.proj(rt)
        rt = self.proj_drop(rt)
        return rt

    def apply_rpe(self, attn, rel_pos):
        if self.use_rpe:
            attn = attn + self.rpe(rel_pos)
        return attn

    def extra_repr(self) -> str:
        return 'dim={}, ct_size={}, num_heads={}'.format(
                    self.dim, self.rt_per_window, self.num_heads)


class OctFormerBlock(torch.nn.Module):
    """
    Octree Transformer Block adapted from https://github.com/octree-nn/octformer,
    with Hierarchical Attention (HAT) design inspired by
    https://github.com/NVlabs/FasterViT.
    """
    
    def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
                 dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 qk_scale: Optional[float] = None, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                 activation: torch.nn.Module = torch.nn.GELU, use_rt: bool = False,
                 rt_size: int = 1, rt_propagation: bool = False,
                 rt_propagation_scale: Optional[float] = None,
                 use_ADaPE: bool = False, disable_RPE: bool = False,
                 conv_norm: str = 'batchnorm', last: bool = False,
                 layer_scale: Optional[float] = None, xcpe: bool = False,
                 **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.use_rt = use_rt
        # self.use_ADaPE = use_ADaPE,
        self.dim = dim
        # NOTE: Dilation is disabled when using carrier tokens, as it is
        #       likely redundant to use both (and carrier tokens for dilated
        #       windows does not make sense).
        dilation = 1 if self.use_rt else dilation
        self.dilated_windows = dilation > 1
        rt_per_window = rt_size if self.use_rt else 0  # track number of carrier tokens per window
        self.rt_per_window = rt_per_window
        self.last = last
        self.rt_propagation = rt_propagation
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        use_rt_propagation_scale = rt_propagation_scale is not None and type(rt_propagation_scale) in [int, float]
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attention = OctreeAttention(dim, patch_size, num_heads, qkv_bias,
                                         qk_scale, attn_drop, proj_drop, dilation,
                                         rt_per_window=rt_per_window,
                                         use_rpe=(not disable_RPE))
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, proj_drop)
        self.drop_path = OctreeDropPath(drop_path, nempty,
                                        dilated_windows=self.dilated_windows)
        self.cpe = CPE(dim, nempty=nempty, conv_norm=conv_norm, xcpe=xcpe)
        # Learnable per-channel scale multiplier, originally proposed by
        # https://arxiv.org/pdf/2103.17239
        self.gamma1 = torch.nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma2 = torch.nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

        if not self.use_rt:  # relay token attention layers
            return
        self.rt_norm1 = torch.nn.LayerNorm(dim)
        self.rt_attention = RTAttentionOld(dim, patch_size, num_heads, qkv_bias,
                                        qk_scale, attn_drop, proj_drop,
                                        rt_per_window,
                                        use_rpe=(not disable_RPE))
        self.rt_norm2 = torch.nn.LayerNorm(dim)
        self.rt_mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, proj_drop)
        self.rt_drop_path = OctreeDropPath(drop_path, nempty, use_ct=True)
        self.rt_gamma1 = torch.nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.rt_gamma2 = torch.nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        
        if not (self.last and self.rt_propagation):
            return
        self.upsampler = torch.nn.Upsample(scale_factor=patch_size//rt_per_window, mode='nearest')
        # Just use a scalar multiplier for RT propagation scaling, which
        # prevents 'blurring' local features with RT features
        self.rt_gamma_propagate = torch.nn.Parameter(torch.tensor(rt_propagation_scale)) if use_rt_propagation_scale else 1

    def forward(self, data: torch.Tensor, relay_tokens: torch.Tensor, octree: OctreeT, depth: int):
        K = self.patch_size
        C = self.dim
        G = self.rt_per_window
        rt = relay_tokens
        
        # Apply conditional positional encoding
        data = data + self.cpe(data, octree, depth)
        # Pad batch and reshape into windows
        data = octree.data_to_windows(  # (N, K, C)
            data, depth, dilated_windows=self.dilated_windows
        )
        # Do global attention via carrier tokens
        if self.use_rt:
            # NOTE: No longer using ADaPE per octformer block
            # if self.use_ADaPE:
            #     rt = rt + self.rt_adape(octree, depth)
            rt_attn = self.rt_gamma1 * self.rt_attention(self.rt_norm1(rt), octree, depth)
            rt = rt + self.rt_drop_path(rt_attn, octree, depth)
            rt_ffn = self.rt_gamma2 * self.rt_mlp(self.rt_norm2(rt))
            rt = rt + self.rt_drop_path(rt_ffn, octree, depth)
            # Concatenate carrier tokens with window tokens
            data = torch.cat((rt.unsqueeze(1), data), dim=1)

        attn = self.gamma1 * self.attention(self.norm1(data), octree, depth)
        data = data + self.drop_path(attn, octree, depth)
        ffn = self.gamma2 * self.mlp(self.norm2(data))
        data = data + self.drop_path(ffn, octree, depth)

        # Split CTs from window tokens
        if self.use_rt:
            rt, data = data.split([G, K], dim=1)
            rt = rt.squeeze(1)
        # Unpad batch and restore original data shape
        data = octree.windows_to_data(
            data, depth, dilated_windows=self.dilated_windows
        )        
        # On last block, propagate carrier token features to local feature map
        if self.last and self.use_rt and self.rt_propagation:
            # TODO: Make this work with ct_size > 1
            mask = octree.rt_init_mask[depth].unsqueeze(-1)
            ct_upsampled = rt.unsqueeze(0).transpose(1, 2)
            ct_upsampled = self.upsampler(ct_upsampled).transpose(1,2).squeeze(0)
            ct_upsampled = ct_upsampled.view(-1, K//G, C)
            # Mask out padded and overlap CTs
            ct_upsampled = ct_upsampled.masked_fill(mask, value=0).view(-1, C)
            ct_upsampled = octree.patch_reverse(ct_upsampled, depth)
            data = data + self.rt_gamma_propagate*ct_upsampled
        return data, rt


class TokenInitialiser(torch.nn.Module):
    """
    Initialises relay tokens by avg pooling over each local attn window.
    """
    
    def __init__(self, dim: int, patch_size: int, nempty: bool, conv_norm: str,
                 rt_size: int = 1, use_cpe: bool = False, xcpe: bool = False):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            patch_size: patch size.
            nempty: only compute on non-empty octree leaf nodes.
            conv_norm: type of normalisation to use after the conv layer.
            rt_size: number of relay tokens per local window.
            use_cpe: disable CPE during token initialisation.
            xcpe: Use xCPE instead of CPE.
        """
        super().__init__()
        self.use_cpe = use_cpe
        if use_cpe:
            self.cpe = CPE(dim, nempty=nempty, conv_norm=conv_norm, xcpe=xcpe)
        # NOTE: Currently, because of how octree windows are constructed,
        #       consecutive batch elements can have an octree window with
        #       elements from both batches. This means avgpooled features for
        #       'leaky' windows will contain features from 2 batch elements, and
        #       not be valid. The only way to prevent this (that I can tell) is
        #       to redo the OCNN batch implementation to include padding around
        #       each batch element. Instead, I opt to ignore 'leaky' window
        #       features during global attention. This should be fine most of
        #       the time as a max of 1 window will be ignored per batch element,
        #       typically out of 100s, but isn't the optimal solution.

        # Pool the features in each octree window, without considering surrounding features
        assert patch_size % rt_size == 0, "Currently, patch_size must be divisible by ct_size"
        # self.pool = torch.nn.AvgPool1d(kernel_size=patch_size//ct_size)
        self.patch_size = patch_size
        self.dim = dim
        self.rt_size = rt_size

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        K = self.patch_size
        C = self.dim
        G = self.rt_size
        
        if self.use_cpe:
            data = self.cpe(data, octree, depth)
        data = octree.patch_partition(data, depth)
        # Reshape to windows, and mask out ignored values as NaN
        # TODO: Make this work with ct_size > 1
        data = data.view(-1, K//G, C)
        mask = octree.rt_init_mask[depth].unsqueeze(-1)
        data = data.masked_fill(mask, value=torch.nan)
        # Avg pool over spatial dimension
        # NOTE: AvgPool1D can't handle NaNs, so use nanmean() instead
        rt = torch.nanmean(data, dim=1)
        assert(not torch.any(rt.isnan())), \
            "NaN propagated during RT init, check code"
        return rt


class OctFormerStage(torch.nn.Module):

    def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
                 dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 qk_scale: Optional[float] = None, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                 activation: torch.nn.Module = torch.nn.GELU, interval: int = 6,
                 disable_RPE: bool = False, use_rt: bool = False, rt_size: int = 1,
                 rt_propagation: bool = False,
                 rt_propagation_scale: Optional[float] = None,
                 ADaPE_mode: Optional[str] = None,
                 grad_checkpoint: bool = True, num_blocks: int = 2,
                 conv_norm: str = 'batchnorm', layer_scale: Optional[float] = None,
                 xcpe: bool = False, octformer_block=OctFormerBlock, **kwargs):
        super().__init__()
        self.num_blocks = num_blocks
        self.grad_checkpoint = grad_checkpoint
        self.use_rt = use_rt
        self.use_ADaPE = ADaPE_mode is not None
        # self.interval = interval  # normalisation interval
        # self.num_norms = (num_blocks - 1) // self.interval

        self.blocks = torch.nn.ModuleList([octformer_block(
            dim=dim, num_heads=num_heads, patch_size=patch_size,
            dilation=1 if (i % 2 == 0) else dilation,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            nempty=nempty, activation=activation, disable_RPE=disable_RPE,
            use_rt=use_rt, rt_size=rt_size, rt_propagation=rt_propagation,
            rt_propagation_scale=rt_propagation_scale, use_ADaPE=self.use_ADaPE,
            conv_norm=conv_norm, last=(i == num_blocks - 1),
            layer_scale=layer_scale, xcpe=xcpe) for i in range(num_blocks)])
        # self.norms = torch.nn.ModuleList([
        #     torch.nn.BatchNorm1d(dim) for _ in range(self.num_norms)])
        if not self.use_rt:
            return
        self.global_tokeniser = TokenInitialiser(dim, patch_size=patch_size,
                                                 nempty=nempty,
                                                 conv_norm=conv_norm,
                                                 rt_size=rt_size,
                                                 use_cpe=(not self.use_ADaPE),
                                                 xcpe=xcpe)
        if self.use_ADaPE:
            self.rt_adape = ADaPE(dim, activation, ADaPE_mode)

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        rt = self.global_tokeniser(data, octree, depth) if self.use_rt else None
        # Inject positional encoding for RTs
        if self.use_ADaPE and rt is not None:
            rt = rt + self.rt_adape(octree, depth)
        for i in range(self.num_blocks):
            if self.grad_checkpoint and self.training:
                data, rt = checkpoint(self.blocks[i], data, rt, octree, depth, use_reentrant=False)  # disable reentrant to fix error with no_grad?
            else:
                data, rt = self.blocks[i](data, rt, octree, depth)
            # if i % self.interval == 0 and i != 0:
            #   data = self.norms[(i - 1) // self.interval](data)
        return data


class PatchEmbed(torch.nn.Module):
    """
    Generate input embeddings with small conv network.
    """
    def __init__(self, in_channels: int = 3, dim: int = 96, num_down: int = 2,
                 nempty: bool = True, downsample_input_embeddings: bool = True,
                 conv_norm: str = 'batchnorm', **kwargs):
        super().__init__()
        self.num_stages = num_down
        self.delta_depth = -num_down
        self.downsample_input_embeddings = downsample_input_embeddings

        if self.downsample_input_embeddings:
            channels = [int(dim * 2**i) for i in range(-self.num_stages, 1)]
            self.convs = torch.nn.ModuleList([OctreeConvNormRelu(
                in_channels if i == 0 else channels[i], channels[i], kernel_size=[3],
                stride=1, nempty=nempty, conv_norm=conv_norm) for i in range(self.num_stages)])
            self.downsamples = torch.nn.ModuleList([OctreeConvNormRelu(
                channels[i], channels[i+1], kernel_size=[2], stride=2, nempty=nempty, conv_norm=conv_norm)
                for i in range(self.num_stages)])
            self.proj = OctreeConvNormRelu(
                channels[-1], dim, kernel_size=[3], stride=1, nempty=nempty, conv_norm=conv_norm)
        else:
            self.convs = torch.nn.ModuleList([OctreeConvNormRelu(
                in_channels if i == 0 else dim, dim, kernel_size=[3],
                stride=1, nempty=nempty, conv_norm=conv_norm) for i in range(self.num_stages)])

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        if self.downsample_input_embeddings:
            for i in range(self.num_stages):
                depth_i = depth - i
                data = self.convs[i](data, octree, depth_i)
                data = self.downsamples[i](data, octree, depth_i)
            data = self.proj(data, octree, depth_i - 1)
        else:
            for i in range(self.num_stages):
                data = self.convs[i](data, octree, depth)
        return data


class Downsample(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: List[int] = [2], nempty: bool = True,
                 conv_norm: str = 'batchnorm'):
        super().__init__()
        self.conv = ocnn.nn.OctreeConv(in_channels, out_channels, kernel_size,
                                    stride=2, nempty=nempty, use_bias=True)
        self.norm = get_norm_layer(out_channels, conv_norm)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.conv(data, octree, depth)
        data = self.norm(data)
        return data