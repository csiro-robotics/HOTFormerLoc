# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
#
# Adapted from https://github.com/octree-nn/octformer
# by Ethan Griffiths.
# --------------------------------------------------------

from typing import Optional, List, Union

import torch
import torch.nn.functional as F
import ocnn
from ocnn.octree import Octree

from misc.utils import rescale_octree_points


def pad_sequence(batch_list, fill_value: int = 0) -> torch.Tensor:
    """
    Collate list of different size tensors into a batch via padding. I found
    this implementation faster than torch.nn.utils.rnn.pad_sequence().
    """
    data_padded_list = []
    max_size = max([row.size(0) for row in batch_list])
    for row in batch_list:
        data_padded_list.append(
            F.pad(
                row, pad=(0, 0, 0, max_size - row.size(0)), value=fill_value
            )
        )
    data_padded = torch.stack(data_padded_list)
    return data_padded


class OctreeT(Octree):
    """
    Octree window attention data structure adapted from
    https://github.com/octree-nn/octformer, with Hierarchical Attention (HAT)
    design inspired by https://github.com/NVlabs/FasterViT.
    """
    def __init__(self, octree: Octree, patch_size: int = 24, dilation: int = 4,
                 nempty: bool = True, max_depth: Optional[int] = None,
                 start_depth: Optional[int] = None,
                 rt_layers: List[bool] = [False, False, False, False],
                 rt_size: int = 0, ADaPE_mode: Optional[str] = None,
                 num_pyramid_levels: int = 0, num_octf_levels: int = 0,
                 **kwargs):
        super().__init__(octree.depth, octree.full_depth)
        self.__dict__.update(octree.__dict__)

        self.patch_size = patch_size
        self.dilation = dilation  # TODO dilation as a list
        self.rt_layers = rt_layers
        self.rt_size = rt_size
        self.nempty = nempty
        self.max_depth = max_depth or self.depth
        self.start_depth = start_depth or self.full_depth
        self.num_pyramid_levels = num_pyramid_levels
        self.num_octf_levels = num_octf_levels
        if self.num_pyramid_levels > 0:  # HOTFormerLoc
            self.rt_layers = [False]*self.num_octf_levels + [True]*self.num_pyramid_levels
            self.pyramid_depths = [(self.max_depth - self.num_octf_levels - j) for j in range(self.num_pyramid_levels)]
        self.invalid_mask_value = -1e3
        self.ADaPE_mode = ADaPE_mode
        self.use_ADaPE = self.ADaPE_mode is not None
        if self.ADaPE_mode == "cov":
            self.cov_idx = torch.triu_indices(3, 3, device=self.device)
        assert self.start_depth >= 1, "Octree not deep enough for model depth"

        self.block_num = patch_size * dilation
        self.nnum_t = self.nnum_nempty if nempty else self.nnum
        self.nnum_a = ((self.nnum_t / self.block_num).ceil() * self.block_num).int()

        num = self.max_depth + 1
        self.batch_idx = [None] * num
        self.hat_batch_window_idx = [None] * num
        self.rt_batch_idx = [None] * num
        self.batch_boundary = [None] * num
        self.batch_num_windows = [None] * num
        self.batch_num_relay_tokens_combined = None
        self.batch_window_overlap_mask = [None] * num
        self.patch_mask = [None] * num
        self.dilate_mask = [None] * num
        self.hat_window_mask = [None] * num
        self.rt_mask = [None] * num
        self.rt_init_mask  = [None] * num
        self.rt_attn_mask = None
        self.rel_pos = [None] * num
        self.dilate_pos = [None] * num
        self.window_stats = [None] * num

    def build_t(self):
        r"""Build the information necessary for computing Octree attention.

        This includes attention masks, relative positions, batch idx, and point
        window distribution stats. This function must be called before passing
        the octree through OctFormer.
        """
        for i, depth in enumerate(range(self.start_depth, self.max_depth + 1)):
            use_rt = self.rt_layers[-(i+1)]
            self.build_batch_idx(depth, use_rt)
            self.build_batch_boundary(depth, use_rt)
            self.build_attn_mask(depth, use_rt)
            self.build_rel_pos(depth)
            self.compute_window_stats(depth, use_rt)
            
        if self.num_pyramid_levels > 0:
            self.build_rt_attn_mask()

    def build_batch_idx(self, depth: int, use_rt: bool):
        # Build batch idx for regular octree operation
        batch = self.batch_id(depth, self.nempty)
        batch = self.patch_partition(batch, depth, self.batch_size)
        self.batch_idx[depth] = batch
        if not use_rt:
            return

        # Build idx for HAT windows, with RT added to local windows
        # NOTE: Currently, overlapping RTs are not masked out, and instead are
        #       masked so that they only attend to features from the leftmost
        #       batch element (i.e. floor of the batch idxs they belong to)
        batch_window = batch.view(-1, self.patch_size)
        batch_rt_idx = batch_window.min(1, keepdim=True).values
        # Save mask for RT initialisation (prevents pooling erroneous features)
        self.rt_init_mask[depth] = batch_window != batch_rt_idx
        # Add RT to mask
        hat_batch_window = F.pad(batch_window, pad=(self.rt_size, 0))
        hat_batch_window[:, :self.rt_size] += batch_rt_idx  # insert RT batch idx
        self.hat_batch_window_idx[depth] = hat_batch_window

        # Build idx for RT global attn
        # TODO: ensure this works with rt_size > 1
        batch_rt = batch.view(-1, self.patch_size // self.rt_size)
        self.rt_batch_idx[depth] = batch_rt.min(1).values

    def build_batch_boundary(self, depth: int, use_rt: bool):
        """
        Get the boundary indices for each batch elem. Useful for separating RTs
        into batches with torch.split().
        """
        if not use_rt:
            return
        batch_nnum_cumsum = self.batch_nnum_nempty[depth].cumsum(0)
        # Add patch partition padding to last elem
        num_padded = self.nnum_a[depth] - self.nnum_t[depth]
        batch_nnum_cumsum[-1] = batch_nnum_cumsum[-1] + num_padded
        # Get idxs where batch changes
        batch_boundary_floor = batch_nnum_cumsum // self.patch_size
        # Get number of leftover points in last window of each batch
        batch_window_remainder = batch_nnum_cumsum % self.patch_size
        # TODO: Remove the next two class variables, as they are only used in
        #       debugging currently.
        # Create mask for batch windows that contain overlapping batch data
        self.batch_window_overlap_mask[depth] = batch_window_remainder.masked_fill(
            batch_window_remainder != 0, 1
        )
        # Correct indices for splitting with tensor_split
        self.batch_boundary[depth] = batch_boundary_floor \
                                     + self.batch_window_overlap_mask[depth]
        # Also get number of windows per batch elem, inclusive of overlap with
        # next elem (used for torch.split)
        self.batch_num_windows[depth] = torch.diff(
            self.batch_boundary[depth], prepend=torch.zeros(1)
        ).int()

    def build_attn_mask(self, depth: int, use_rt: bool):
        """
        Compute attention masks for window attention and relay token attention,
        so that attention ignores padding and all RTs that contain neighbouring
        batch information.
        """
        # Window and dilation masks
        batch = self.batch_idx[depth]
        mask = batch.view(-1, self.patch_size)
        self.patch_mask[depth] = self._calc_attn_mask(mask)

        mask = batch.view(-1, self.patch_size, self.dilation)
        mask = mask.transpose(1, 2).reshape(-1, self.patch_size)
        self.dilate_mask[depth] = self._calc_attn_mask(mask)

        # Window + RT mask (HAT)
        if not use_rt:
            return
        # NOTE: Currently, overlapping RTs are not masked out, and instead are
        #       masked so that they only attend to features from the leftmost
        #       batch element they belong to
        # TODO: check this works with ct_size > 1
        mask = self.hat_batch_window_idx[depth]
        self.hat_window_mask[depth] = self._calc_attn_mask(mask)

        # RT Mask
        batch_num_windows_list = self.batch_num_windows[depth].tolist()
        mask = batch.view(-1, self.patch_size)
        mask_split = mask.split(batch_num_windows_list)
        # Pad with values higher than batch size will ever be, to ensure fill
        # overrides batch idx per window
        mask_padded = pad_sequence(
            mask_split, fill_value=(self.batch_size + 1e4)
        )
        # Use left-most batch idx for carrier tokens
        mask_padded = mask_padded.min(dim=2).values
        self.rt_mask[depth] = self._calc_attn_mask(mask_padded)

        # # TODO: Below is the start of code to correct mask for overlap windows,
        # #       so that batch element with higher overlap is unmasked
        # mode_max_mask = (batch_windows.max(dim=1).values == batch_windows.mode(dim=1).values)
        # torch.nonzero(mode_max_mask == False)
    
    def build_rt_attn_mask(self):
        """
        Compute the attention mask for multi-scale relay tokens.
        """
        for j, depth_j in enumerate(self.pyramid_depths):
            batch_num_relay_tokens_depth_j = self.batch_num_windows[depth_j]
            if j == 0:
                batch_num_relay_tokens_combined = batch_num_relay_tokens_depth_j.clone().detach()
            else:
                batch_num_relay_tokens_combined += batch_num_relay_tokens_depth_j
        self.batch_num_relay_tokens_combined = batch_num_relay_tokens_combined

        # Generate (B, N) mask of batch idx for all relay tokens.
        # (N = number of relay tokens in largest batch element, which all
        #   elements are padded to)
        B = self.batch_size
        N = self.batch_num_relay_tokens_combined.max().item()
        rt_batch_idx = torch.full(
            (B, N), fill_value = 1e4, dtype=torch.long, device=self.device
        )
        for batch_idx, batch_length in enumerate(self.batch_num_relay_tokens_combined):
            rt_batch_idx[batch_idx, :batch_length] = batch_idx

        # Correct the mask for last batch (as padding is unaccounted for)
        prev_end_idx = 0
        for depth_j in self.pyramid_depths:
            num_padded_tokens = torch.sum(self.rt_batch_idx[depth_j]
                                          >= B).item()
            batch_rel_idx = self.batch_num_windows[depth_j][-1].item()
            if num_padded_tokens > 0:
                batch_end_idx = prev_end_idx + batch_rel_idx
                batch_pad_start_idx = batch_end_idx - num_padded_tokens
                rt_batch_idx[-1, batch_pad_start_idx:batch_end_idx] = B
            prev_end_idx += batch_rel_idx

        # Convert to attn mask with (B, N, N)
        self.rt_attn_mask = self._calc_attn_mask(rt_batch_idx)

    def _calc_attn_mask(self, mask: torch.Tensor):
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, self.invalid_mask_value)
        return attn_mask

    def build_rel_pos(self, depth: int):
        key = self.key(depth, self.nempty)
        key = self.patch_partition(key, depth)
        x, y, z, _ = ocnn.octree.key2xyz(key, depth)
        xyz = torch.stack([x, y, z], dim=1)

        xyz = xyz.view(-1, self.patch_size, 3)
        self.rel_pos[depth] = xyz.unsqueeze(2) - xyz.unsqueeze(1)

        xyz = xyz.view(-1, self.patch_size, self.dilation, 3)
        xyz = xyz.transpose(1, 2).reshape(-1, self.patch_size, 3)
        self.dilate_pos[depth] = xyz.unsqueeze(2) - xyz.unsqueeze(1)

    def compute_window_stats(self, depth: int, use_rt: bool):
        """
        Pre-compute mean and covariance of each point window. Used to enhance
        positional encoding (ADaPE) learned by MLP for carrier token attention.
        """
        if not use_rt or not self.use_ADaPE:
            return
        N = self.nnum_a[depth] // self.patch_size
        assert self.ADaPE_mode in ['pos','var','cov'], "Invalid mode provided"
        mode_num_feat_dict = {'pos': 3, 'var': 6, 'cov': 9}
        # Num feats = 3 (μx,μy,μz) + 6 (upper tri of cov matrix: σx, σxy, σxz, σy, σyz, σz)
        C = mode_num_feat_dict[self.ADaPE_mode]
        # Get points for current depth
        x, y, z, _ = self.xyzb(depth, self.nempty)
        points = torch.stack((x,y,z), dim=1).to(torch.float32)
        # Rescale to [-1, 1] and put into windows
        points = rescale_octree_points(points, depth)
        points = self.data_to_windows(points, depth, dilated_windows=False)
        mask = self.rt_init_mask[depth]
        window_stats = torch.zeros(N, C, device=self.device)
        # Compute (μx, μy, μz, σx, σxy, σxz, σy, σyz, σz) for all windows
        for i, window_points in enumerate(points):
            # Mask out points from overlap windows
            batch_masked = window_points[~mask[i]]
            window_stats[i,:3] = batch_masked.mean(0)
            # NOTE: Currently, windows with only 1 unmasked point are assumed
            #       to have covariance matrix of zeros. Better solution is
            #       to ensure point windows are assigned to the batch submap
            #       that they contain the most of, but this requires extra
            #       masking logic and currently isn't worth fixing.
            if self.ADaPE_mode == 'var':
                if batch_masked.size(0) < 2:
                    window_stats[i,3:] = torch.zeros(1, 3, device=self.device,
                                          dtype=torch.float32)
                else:
                    window_stats[i,3:] = batch_masked.var(0)
            elif self.ADaPE_mode == 'cov':
                if batch_masked.size(0) < 2:
                    cov = torch.zeros(3, 3, device=self.device,
                                          dtype=torch.float32)
                else:
                    cov = batch_masked.T.cov()
                window_stats[i,3:] = cov[self.cov_idx[0], self.cov_idx[1]]
            
        assert(not torch.any(window_stats.isnan())), \
            "NaN propagated during window stats computation, check code"
        self.window_stats[depth] = window_stats        

    def patch_partition(self, data: torch.Tensor, depth: int, fill_value=0):
        num = self.nnum_a[depth] - self.nnum_t[depth]
        tail = data.new_full((num,) + data.shape[1:], fill_value)
        return torch.cat([data, tail], dim=0)

    def patch_reverse(self, data: torch.Tensor, depth: int):
        return data[:self.nnum_t[depth]]

    def data_to_windows(self, data: torch.Tensor, depth: int,
                        dilated_windows: bool, fill_value=0):
        """
        Reshape octree data into windows. This function applies padding and
        dilation, so just pass the octree features, depth, and whether dilated
        windows should be used.

        Args:
            data (Tensor): Octree data, which must have shape (N, C)
        """
        C = data.size(-1)
        data = self.patch_partition(data, depth, fill_value)  # (N*K, C)
        if dilated_windows:  # account for dilation
            data = data.view(-1, self.patch_size,
                             self.dilation, C).transpose(1, 2).reshape(-1, C)
        return data.view(-1, self.patch_size, C)  # (N, K, C)

    def windows_to_data(self, data: torch.Tensor, depth: int,
                        dilated_windows: bool):
        """
        Reshape octree windows back into original shape. This function accounts
        for padding and dilation, so just pass the octree features, depth, and
        whether dilated windows are used.

        Args:
            data (Tensor): Octree window data, which must have shape (N, K, C)
        """
        C = data.size(-1)
        data = data.reshape(-1, C)  # (N*K, C)
        if dilated_windows:  # account for dilation
            data = data.view(-1, self.dilation,
                             self.patch_size, C).transpose(1, 2).reshape(-1, C)
        return self.patch_reverse(data, depth)
    
    def to(self, device: Union[torch.device, str], non_blocking: bool = False):
        r""" Moves the octree to a specified device. Adapted from `ocnn.octree`.

        Args:
            device (torch.device or str): The destination device.
            non_blocking (bool): If True and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect. Default: False.
        """
        if isinstance(device, str):
            device = torch.device(device)
            
        #  If on the save device, directly retrun self
        if self.device == device:
            return self

        # Initialise base octree class
        octree = super().to(device, non_blocking)

        def list_to_device(prop):
            return [p.to(device, non_blocking=non_blocking)
                    if isinstance(p, torch.Tensor) else None for p in prop]

        def list_clone(prop):
            return [p.clone()
                    if isinstance(p, torch.Tensor) else None for p in prop]

        # Construct new OctreeT and copy objects over
        octree = OctreeT(octree, self.patch_size, self.dilation, self.nempty,
                         self.max_depth, self.start_depth, self.rt_layers,
                         self.rt_size, self.ADaPE_mode)
        octree.batch_idx = list_to_device(self.batch_idx)
        octree.hat_batch_window_idx = list_to_device(self.hat_batch_window_idx)
        octree.rt_batch_idx = list_to_device(self.rt_batch_idx)
        octree.batch_boundary = list_clone(self.batch_boundary)  # CPU
        octree.batch_num_windows = list_clone(self.batch_num_windows)  # CPU
        octree.batch_window_overlap_mask = list_clone(self.batch_window_overlap_mask)  # CPU
        octree.patch_mask = list_to_device(self.patch_mask)
        octree.dilate_mask = list_to_device(self.dilate_mask)
        octree.hat_window_mask = list_to_device(self.hat_window_mask)
        octree.rt_mask = list_to_device(self.rt_mask)
        octree.rt_init_mask  = list_to_device(self.rt_init_mask)
        octree.rel_pos = list_to_device(self.rel_pos)
        octree.dilate_pos = list_to_device(self.dilate_pos)
        octree.window_stats = list_to_device(self.window_stats)        
        return octree