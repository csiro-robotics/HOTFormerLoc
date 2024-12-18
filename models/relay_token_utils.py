"""
Utils for processing multi-scale relay tokens in batches.

Adapted by Ethan Griffiths (Data61, Pullenvale)
"""
import torch
from torch.nn.utils.rnn import unpad_sequence

from typing import Optional, List, Tuple, Dict
from models.octree import OctreeT, pad_sequence

def concat_and_pad_rt(
    relay_token_dict: Dict[int, torch.Tensor],
    octree: OctreeT,
    pyramid_depths: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenates relay tokens from different levels in the pyramid
    batch-wise, then applies padding for parallelisation. Returns a single
    (B, N, C) tensor, where B = batch size, N = number of tokens (incl.
    padding), and C = channel size.
    """
    if pyramid_depths is None:
        pyramid_depths = list(relay_token_dict.keys())
    # Split relay tokens into batches for each depth
    relay_tokens_split_per_depth = []
    for depth_j in pyramid_depths:
        batch_num_relay_tokens_depth_j = octree.batch_num_windows[depth_j]
        relay_tokens_split_per_depth.append(
            relay_token_dict[depth_j].split(batch_num_relay_tokens_depth_j.tolist())
        )
    
    # Combine relay tokens for each batch in all depths
    relay_tokens_combined_list = []
    for relay_token_pyramid_batch in zip(*relay_tokens_split_per_depth):
        relay_tokens_combined_list.append(
            torch.cat(relay_token_pyramid_batch)
        )
    padded_pyramid_relay_tokens = pad_sequence(relay_tokens_combined_list)
    return padded_pyramid_relay_tokens

def unpad_and_split_rt(
    padded_pyramid_relay_tokens: torch.Tensor,
    octree: OctreeT,
    pyramid_depths: List[int],
) -> dict[int, torch.Tensor]:
    """
    Reverses the concatenation and padding applied to multi-scale relay
    tokens. Returns a dictionary where keys are octree depth, and values
    are the corresponding relay tokens in a (M, C) tensor.
    """
    # Remove padding
    relay_tokens_combined_list = unpad_sequence(
        padded_pyramid_relay_tokens, octree.batch_num_relay_tokens_combined,
        batch_first=True
    )
    # Separate relay tokens for each depth
    batch_num_relay_tokens_per_depth = [
        octree.batch_num_windows[depth_j].tolist()
            for depth_j in pyramid_depths
    ]
    relay_tokens_split_per_depth = [
        [] for _ in range(len(pyramid_depths))
    ]
    for i, batch_num_tokens in enumerate(zip(*batch_num_relay_tokens_per_depth)):
        relay_tokens_split_temp = relay_tokens_combined_list[i].split(
            batch_num_tokens
        )
        for j in range(len(pyramid_depths)):
            relay_tokens_split_per_depth[j].append(
                relay_tokens_split_temp[j]
            )

    # Concatenate relay tokens for each depth and put back in dict
    relay_token_dict = {}
    for i, depth_j in enumerate(pyramid_depths):
        relay_token_dict[depth_j] = torch.cat(relay_tokens_split_per_depth[i])
    
    return relay_token_dict