"""
Utility functions for converting to and from cylindrical/cartesian coordinate
systems.

By Ethan Griffiths (Data61, Pullenvale)
"""

from abc import ABC, abstractmethod

import numpy as np
import torch


class CoordinateSystem(ABC):
    """
    Base class to handle converting to and from different coordinate systems. 
    """
    def __init__(self, use_octree: bool):
        """
        Args:
            use_octree (bool): Whether octrees are being used (to convert
                outputs into correct range)
        """
        self.use_octree = use_octree

    @abstractmethod
    def __call__(self, pc):
        pass

    def cartesian_to_cylindrical(self, pc: torch.Tensor):
        """
        Args:
            pc (torch.Tensor): (N, 3) point cloud with Cartesian coordinates
                (x, y, z)
        Returns:
            cylindrical_pc (torch.Tensor): (N, 3) point cloud in (ρ, φ, z)
                coordinates, where ρ is radial distance, φ is heading angle
                [rad], and z is height
        """
        # phi is the azimuth angle in radians in [-pi, pi] range
        phi = torch.atan2(pc[:, 1], pc[:, 0])
        # rho is a distance from a coordinate origin
        rho = torch.sqrt(pc[:, 0]**2 + pc[:, 1]**2)
        z = pc[:, 2]
        cylindrical_pc = torch.stack([rho, phi, z], dim=1)
        return cylindrical_pc
    
    def cylindrical_to_cartesian(self, pc: torch.Tensor):
        """
        Args:
            cylindrical_pc (torch.Tensor): (N, 3) point cloud with Cylindrical
                coordinates (ρ, φ, z), where ρ is radial distance, φ is heading
                angle [rad], and z is height
        Returns:
            pc (torch.Tensor): (N, 3) point cloud in (x, y, z)
        """
        x = pc[:, 0] * torch.cos(pc[:, 1])
        y = pc[:, 0] * torch.sin(pc[:, 1])
        z = pc[:, 2]
        
        cartesian_pc = torch.stack([x, y, z], dim=1)
        return cartesian_pc

class CylindricalCoordinates(CoordinateSystem):
    def __init__(self, use_octree: bool, *args, **kwargs):
        super().__init__(use_octree, *args, **kwargs)
        
    def __call__(self, pc: torch.Tensor):
        """
        Converts point cloud to cylindrical coordinates and returns it.
        Args:
            pc (torch.Tensor): (N, 3) point cloud with Cartesian coordinates
                (x, y, z)

        Returns:
            cylindrical_pc (torch.Tensor): (N, 3) point cloud in (ρ, φ, z)
                coordinates, where ρ is radial distance, φ is heading angle
                [rad], and z is height
        """
        assert pc.ndim == 2
        assert pc.shape[1] == 3
        # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
        assert torch.all(abs(pc) <= 1.0)
        cylindrical_pc = self.cartesian_to_cylindrical(pc)
        
        # Scale coordinates to [-1, 1] for octree construction
        if self.use_octree:
            cylindrical_pc = self.scale_coords(cylindrical_pc)            
            # Final sanity check for value range
            cylindrical_pc = torch.clamp(cylindrical_pc, -1.0, 1.0)
        return cylindrical_pc

    def undo_conversion(self, pc: torch.Tensor):
        """
        Undo the operation done when this class is called.
        """
        assert pc.ndim == 2
        assert pc.shape[1] == 3
        if self.use_octree:
            pc = self.unscale_coords(pc)
        cartesian_pc = self.cylindrical_to_cartesian(pc)
        return cartesian_pc        

    def scale_coords(self, pc: torch.Tensor):
        """
        Scale cylindrical coordinates to be within [-1, 1]
        """
        assert pc.ndim == 2
        assert pc.shape[1] == 3
        rho = pc[:, 0].numpy()
        phi = pc[:, 1].numpy()
        rho_scaled = torch.tensor(np.interp(rho, [0, 1], [-1, 1]))
        phi_scaled = torch.tensor(np.interp(phi, [-np.pi, np.pi], [-1, 1]))
        pc[:, 0] = rho_scaled
        pc[:, 1] = phi_scaled        
        return pc
    
    def unscale_coords(self, pc: torch.Tensor):
        """
        Rescale octree cylindrical coordinates to be within rho: [0, 1] and
        phi: [-pi, pi]
        """
        assert pc.ndim == 2
        assert pc.shape[1] == 3
        rho_scaled = pc[:, 0].numpy()
        phi_scaled = pc[:, 1].numpy()
        rho = torch.tensor(np.interp(rho_scaled, [-1, 1], [0, 1]))
        phi = torch.tensor(np.interp(phi_scaled, [-1, 1], [-np.pi, np.pi]))
        pc[:, 0] = rho
        pc[:, 1] = phi
        return pc

    
class CartesianCoordinates(CoordinateSystem):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __call__(self, pc):
        return pc