"""
Code for SALSA's adaptive attentional pooling, and MLP token mixer:
https://arxiv.org/pdf/2407.08260.

Adapted by Ethan Griffiths (Data61, Pullenvale)
"""

import torch
from torch import nn
from torch.nn import functional as F

class AdaptivePooling(nn.Module):
    def __init__(self, feature_dim: int, k_pooled_tokens: int):
        """
        Args:
            feature_dim (int): Feature dim size
            k_pooled_tokens (int): Number of tokens to pool to            
        """
        super().__init__()
        self.k_pooled_tokens = k_pooled_tokens
        self.query = nn.Parameter(torch.randn(k_pooled_tokens, feature_dim))
        self.scale = feature_dim ** -0.5
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, attn_mask = None, return_weights=False):
        """
        Args:
            x: Input tensor of shape (batch_size, num_input_tokens, feature_dim)

        Returns:
            Output tensor of shape (batch_size, k_pooled_tokens, feature_dim)
        """
        B, _, C = x.shape
        query = self.query.unsqueeze(0).repeat(x.shape[0],1,1)
        if attn_mask is not None:
            attn_mask = attn_mask.to(query.dtype)
        
        if torch.__version__ >= torch.torch_version.TorchVersion(2.0):
            out = F.scaled_dot_product_attention(
                query=query, key=x, value=x, attn_mask=attn_mask
            )
        else:
            query = query * self.scale
            # attn
            attn = query @ x.transpose(-2, -1)        # (B, H, N, N)
            attn = attn + attn_mask
            attn = self.softmax(attn)
            out = (attn @ x)  # (B, K, C)
        
        if return_weights:
            attn_scores = torch.einsum('ij,bkj->bik', self.query, x)
            attn_weights = F.softmax(attn_scores, dim=1)
            return out, attn_weights

        return out

    
class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1, activation=nn.GELU):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            activation(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class Mixer(nn.Module):
    def __init__(self,
                 k_input_tokens=35000,
                 k_output_tokens=1000,
                 in_d=30,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_d=4,
                 ) -> None:
        super().__init__()

        self.in_d = in_d

        self.out_d = out_d # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=in_d, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.row_proj = nn.Linear(in_d, out_d)
        self.channel_proj = nn.Linear(k_input_tokens, k_output_tokens)

    def forward(self, x: torch.Tensor):
        # x = x.unsqueeze(0)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = x.flatten(1)
        return x