# Pooling methods code based on: https://github.com/filipradenovic/cnnimageretrieval-pytorch

from typing import Dict, List, Union
from collections.abc import Sequence
import torch
import torch.nn as nn
from torch import Tensor
import ocnn
from ocnn.octree import Octree
from models.octree import OctreeT, pad_sequence

from models.layers.netvlad import NetVLADLoupe, GatingContext
from models.layers.salsa import AdaptivePooling, Mixer
from models.layers.octformer_layers import MLP
from models.relay_token_utils import concat_and_pad_rt


class OctGeM(nn.Module):
    """
    Octree compatible version of GeM pooling.
    """
    def __init__(self, input_dim, p=3, eps=1e-6):
        super(OctGeM, self).__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = ocnn.nn.OctreeGlobalPool(nempty=True)

    def forward(self, x: Union[Tensor, Dict[int, Tensor]],
                octree: Octree, depth: int):
        if isinstance(x, dict):
            # HOTFormerLoc, use finest granularity for pooling (max depth)
            depth, x = max(x.items())
        # This implicitly applies ReLU on x (clamps negative values)
        temp = x.clamp(min=self.eps).pow(self.p)
        temp = self.f(temp, octree, depth)  # Apply GlobalAvgPooling
        return temp.pow(1./self.p)          # Return (batch_size, n_features) tensor


class RelayTokenGeM(OctGeM):
    """
    GeM pooling compatible with multi-scale relay tokens (or really any
    batched tensor input.)
    """
    def __init__(self, input_dim, p=3, eps=1e-6):
        super().__init__(input_dim=input_dim, p=p, eps=eps)
        self.f = None
    
    def forward(self, x: Tensor):  # x: (B, N, C)
        # This implicitly applies ReLU on x (clamps negative values)
        temp = x.clamp(min=self.eps).pow(self.p)
        temp = torch.mean(temp, dim=1)  # Apply GlobalAvgPooling
        return temp.pow(1./self.p)      # Return (batch_size, n_features) tensor


class PyramidOctGeMWrapper(nn.Module):
    def __init__(self, input_dim, output_dim, num_pyramid_levels: int,
                 channels: List[int], p=3, eps=1e-6, gating=False,
                 add_batch_norm=True):
        super().__init__()
        self.input_dim = input_dim
        assert num_pyramid_levels > 0, "Minimum 1 pyramid layer"
        self.num_pyramid_levels = num_pyramid_levels
        if len(channels) == 1:
            input_concat_dim = input_dim*num_pyramid_levels
        else:
            assert len(channels) == num_pyramid_levels, "Incorrect num channels"
            input_concat_dim = sum(channels)
        # Same output number of channels as input number of channels
        self.output_dim = output_dim
        self.p = nn.Parameter(torch.ones(num_pyramid_levels) * p)
        self.eps = eps
        self.gating = gating
        self.f = ocnn.nn.OctreeGlobalPool(nempty=True)
        self.linear_bn = nn.Sequential(
            nn.Linear(
                input_concat_dim, output_dim, bias=False
            ),
            nn.BatchNorm1d(input_dim),
        )
        if self.gating:
            self.context_gating = GatingContext(output_dim,
                                                add_batch_norm=add_batch_norm)

    def forward(self, local_feat_dict: Dict[int, Tensor], octree: OctreeT,
                depth: int = None):
        # Generate global descriptor for each pyramid level
        pyramid_descriptors = []
        for j, depth_j in enumerate(local_feat_dict.keys()):
            temp = local_feat_dict[depth_j].clamp(min=self.eps).pow(self.p[j])
            temp = self.f(temp, octree, depth_j)  # Apply GlobalAvgPooling
            pyramid_descriptors.append(temp.pow(1./self.p[j]))  # (batch_size, n_features) tensor
        
        # Concat and fuse into a single descriptor
        global_descriptor = torch.cat(pyramid_descriptors, dim=-1)
        global_descriptor = self.linear_bn(global_descriptor)

        if self.gating:
            global_descriptor = self.context_gating(global_descriptor)
        
        return global_descriptor


class PyramidAttnPoolWrapper(nn.Module):
    """
    Wrapper for adaptive attention pooling + MLP token mixer on pyramidal
    local features, inspired by SALSA: https://arxiv.org/pdf/2407.08260.
    Also allows using GeM instead of token mixer.
    """
    def __init__(self, feature_size: int = 256, output_dim: int = 256,
                 channels: List[int] = [256], num_pyramid_levels: int = 3,
                 k_pooled_tokens: List[int] = [74, 36, 18],
                 mlp_ratio: int = 1, aggregator: str = 'mixer',
                 mix_depth: int = 4):
        super().__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim 
        self.use_projections = True      
        if len(channels) == 1:
            channels = channels * num_pyramid_levels
            self.use_projections = False
        else:
            assert len(channels) == num_pyramid_levels, "Incorrect num channels"
            channels = channels
        self.channels = channels
        self.num_pyramid_levels = num_pyramid_levels
        assert (isinstance(k_pooled_tokens, Sequence)
                and len(k_pooled_tokens) == num_pyramid_levels), (
                    "k_pooled_tokens must be list of k for each pyramid level"
                )
        self.k_pooled_tokens = k_pooled_tokens
        self.k_pooled_tokens_total = sum(k_pooled_tokens)
        self.mlp_ratio = mlp_ratio
        self.aggregator = aggregator
        self.attpool = torch.nn.ModuleList([AdaptivePooling(
            feature_dim=channels[j],             # originally 512
            k_pooled_tokens=k_pooled_tokens[j],  # originally 16
        ) for j in range(num_pyramid_levels)])
        if self.use_projections:
            self.local_projections = torch.nn.ModuleList([])
            for j in range(num_pyramid_levels):
                if channels[j] != feature_size:
                    self.local_projections.append(torch.nn.Linear(
                        in_features=channels[j],
                        out_features=feature_size,
                    ))
                else:
                    self.local_projections.append(torch.nn.Identity())
        
        if aggregator.lower() == 'mixer':
            # TODO: Currently these values are based on a fixed ratio to ensure
            #       the output is equal to output_dim, but may be worth trying
            #       different ratios of tokens to channels in the MLP mixer.
            k_output_tokens = self.k_pooled_tokens_total // 4  # originally 128
            out_d = output_dim // k_output_tokens   # originally 4
            assert k_output_tokens * out_d == output_dim, (
                f"Invalid k for k_pooled_tokens: {k_pooled_tokens}, "
                + f"not compatible with output dim {output_dim}"
            )
            self.descriptor_extractor = Mixer(
                k_input_tokens=self.k_pooled_tokens_total,
                k_output_tokens=k_output_tokens,
                in_d=feature_size,
                mix_depth=mix_depth,
                mlp_ratio=mlp_ratio,
                out_d=out_d,
            )  # output size = k_output_tokens * out_d
        elif aggregator.lower() == 'gem':
            self.token_processor = nn.Sequential(
                nn.LayerNorm(feature_size),
                MLP(
                    in_features=feature_size,
                    hidden_features=feature_size*mlp_ratio,
                    out_features=output_dim,
                ),
            )
            self.descriptor_extractor = RelayTokenGeM(input_dim=feature_size)
        else:
            raise NotImplementedError(f'No valid aggregator: {aggregator}')

    def forward(self, local_feat_dict: Dict[int, Tensor],
                octree: OctreeT, depth: int = None):
        pyramid_depths = list(local_feat_dict.keys())
        # Compute attentional pooling for local features of each pyramid level
        token_attn = []
        for j, depth_j in enumerate(pyramid_depths):
            # Concat and pad local feats per batch
            local_tokens = self.concat_and_pad_local_feat(
                local_feat_dict[depth_j], octree, depth_j
            )
            attn_mask = self.calc_local_attn_mask(local_tokens, octree, j,
                                                  depth_j)
            # Pool to k tokens
            token_attn_j = self.attpool[j](local_tokens, attn_mask)
            if self.use_projections:
                # Match token dimensions
                token_attn_j = self.local_projections[j](token_attn_j)
            token_attn.append(token_attn_j)
        # Merge tokens from all three levels
        token_attn = torch.cat(token_attn, dim=1)  # (B, N1+N2+N3, C)
        # Aggregate tokens into a global descriptor
        if self.aggregator.lower() != 'mixer':
            token_attn = token_attn + self.token_processor(token_attn)
        global_descriptor = self.descriptor_extractor(token_attn)
        return global_descriptor

    def concat_and_pad_local_feat(self, local_feats, octree, depth_j):
        batch_counts = octree.batch_nnum_nempty[depth_j].tolist()
        local_tokens = pad_sequence(local_feats.split(batch_counts))        
        return local_tokens
    
    def calc_local_attn_mask(self, local_tokens: Tensor, octree: OctreeT,
                             j: int, depth_j: int) -> Tensor:
        """
        Computes attention mask for local feature attentional pooling.
        """
        # Generate (B, N) mask of batch idx for all local tokens.
        # (N = number of largest batch element, which all are padded to reach)
        B, N, C = local_tokens.shape        
        batch_counts = octree.batch_nnum_nempty[depth_j].tolist()
        local_batch_idx = torch.full(
            (B, N), fill_value = 1e4, dtype=torch.long, device=octree.device
        )
        for batch_idx, batch_length in enumerate(batch_counts):
            local_batch_idx[batch_idx, :batch_length] = batch_idx

        attn_mask = octree._calc_attn_mask(local_batch_idx)        
        # All query tokens should ignore padding tokens
        attn_mask = attn_mask[:, 0, :].unsqueeze(1)  # (B, N, N) -> (B, k, N)
        attn_mask = attn_mask.repeat(1, self.k_pooled_tokens[j], 1)
        return attn_mask

class AttnPoolWrapper(nn.Module):
    """
    Wrapper for adaptive attention pooling + MLP token mixer, inspired by
    SALSA: https://arxiv.org/pdf/2407.08260. Also allows using GeM instead of
    token mixer.
    """
    def __init__(self, feature_size: int = 256, output_dim: int = 256,
                 k_pooled_tokens: int = 64, mlp_ratio: int = 1,
                 aggregator: str = 'mixer', mix_depth: int = 4):
        super().__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        assert isinstance(k_pooled_tokens, int), (
            "Only 1 value allowed for k_pooled_tokens when using relay tokens"
        )
        self.k_pooled_tokens = k_pooled_tokens
        self.mlp_ratio = mlp_ratio
        self.aggregator = aggregator
        self.attpool = AdaptivePooling(
            feature_dim=feature_size,         # originally 512
            k_pooled_tokens=k_pooled_tokens,  # originally 16
        )
        if aggregator.lower() == 'mixer':
            # TODO: Currently these values are based on a fixed ratio to ensure
            #       the output is equal to output_dim, but may be worth trying
            #       different ratios of tokens to channels in the MLP mixer.
            k_output_tokens = k_pooled_tokens // 4  # originally 128
            out_d = output_dim // k_output_tokens   # originally 4
            self.descriptor_extractor = Mixer(
                k_input_tokens=k_pooled_tokens,
                k_output_tokens=k_output_tokens,
                in_d=feature_size,
                mix_depth=mix_depth,
                mlp_ratio=mlp_ratio,
                out_d=out_d,
            )  # output size = k_output_tokens * out_d
        elif aggregator.lower() == 'gem':
            self.token_processor = nn.Sequential(
                nn.LayerNorm(feature_size),
                MLP(
                    in_features=feature_size,
                    hidden_features=feature_size*mlp_ratio,
                    out_features=output_dim,
                ),
            )
            self.descriptor_extractor = RelayTokenGeM(input_dim=feature_size)
        else:
            raise NotImplementedError(f'No valid aggregator: {aggregator}')
            

    def forward(self, relay_token_dict: Dict[int, Tensor],
                octree: OctreeT, depth: int = None):
        split_tokens = concat_and_pad_rt(relay_token_dict, octree)
        attn_mask = self.calc_rt_attn_mask(octree.rt_attn_mask)
        # Pool to k tokens
        token_attn = self.attpool(split_tokens, attn_mask)
        # Aggregate tokens into a global descriptor
        if self.aggregator.lower() != 'mixer':
            token_attn = token_attn + self.token_processor(token_attn)
        global_descriptor = self.descriptor_extractor(token_attn)
        return global_descriptor

    def calc_rt_attn_mask(self, rt_attn_mask: Tensor) -> Tensor:
        """
        Alters relay token attention mask to be suitable for
        attentional pooling with learnable query matrix.
        """
        # All query tokens should ignore padding tokens
        attn_mask = rt_attn_mask[:, 0, :].unsqueeze(1)  # (B, N, N) -> (B, k, N)
        attn_mask = attn_mask.repeat(1, self.k_pooled_tokens, 1)
        return attn_mask