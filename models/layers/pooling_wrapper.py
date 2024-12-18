from typing import Union, Dict, Optional, List

from torch import Tensor
import torch.nn as nn

from models.layers.pooling import (
    OctGeM, AttnPoolWrapper, PyramidAttnPoolWrapper, PyramidOctGeMWrapper,
)


class PoolingWrapper(nn.Module):
    def __init__(
        self,
        pool_method: str,
        in_dim: int,
        output_dim: int,
        num_pyramid_levels: Optional[int]=None,
        channels: Optional[List[int]]=None,
        k_pooled_tokens: Union[List[int],int,None]=None,
    ):
        super().__init__()

        self.pool_method = pool_method
        self.in_dim = in_dim
        self.output_dim = output_dim
        self.num_pyramid_levels = num_pyramid_levels
        self.channels = channels
        self.k_pooled_tokens = k_pooled_tokens
        self.pooled_feats = 'local'  # flag if local feats or relay tokens are pooled

        if pool_method == 'OctGeM':
            # Generalized mean pooling (octree-based)
            assert in_dim == output_dim
            self.pooling = OctGeM(input_dim=in_dim)
        elif self.pool_method == 'PyramidNetVLAD':
            raise NotImplementedError(f'Not implemented yet: {self.pool_method}')
        elif self.pool_method == 'PyramidOctGeM':
            # Pyramid GeM pooling using Octree-based implementation
            self.pooling = PyramidOctGeMWrapper(
                input_dim=in_dim, output_dim=output_dim, channels=channels,
                num_pyramid_levels=num_pyramid_levels, gating=False
            )
        elif self.pool_method == 'PyramidOctGeMgc':
            # Pyramid GeM pooling using Octree-based implementation with gating context
            self.pooling = PyramidOctGeMWrapper(
                input_dim=in_dim, output_dim=output_dim, channels=channels,
                num_pyramid_levels=num_pyramid_levels, gating=True
            )
        elif self.pool_method == 'PyramidAttnPoolMixer':
            # Pyramid attentional pooling with token mixing MLP - local features
            self.pooling = PyramidAttnPoolWrapper(
                feature_size=in_dim, output_dim=output_dim, channels=channels,
                num_pyramid_levels=num_pyramid_levels,
                k_pooled_tokens=k_pooled_tokens, aggregator='mixer',
            )
        elif self.pool_method == 'AttnPoolMixer':
            # Attentional pooling with token mixing MLP - relay tokens
            self.pooled_feats = 'relaytokens'
            self.pooling = AttnPoolWrapper(
                feature_size=in_dim, output_dim=output_dim,
                k_pooled_tokens=k_pooled_tokens, aggregator='mixer',
            )
        elif self.pool_method == 'AttnPoolGeM':
            # Attentional pooling with GeM pooling - relay tokens
            self.pooled_feats = 'relaytokens'
            self.pooling = AttnPoolWrapper(
                feature_size=in_dim, output_dim=output_dim,
                k_pooled_tokens=k_pooled_tokens, aggregator='GeM',
            )
        else:
            raise NotImplementedError('Unknown pooling method: {}'.format(pool_method))

    def forward(self, x: Union[Tensor, Dict], octree=None, depth=None):
        if octree is None:
            return self.pooling(x)
        else:
            return self.pooling(x, octree, depth)
