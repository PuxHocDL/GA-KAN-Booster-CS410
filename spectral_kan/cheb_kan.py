"""
ChebKAN: Full Chebyshev Kolmogorov-Arnold Network model.

A multi-layer KAN using Chebyshev polynomial activations on each edge.
Includes input normalization to [-1, 1] domain required by Chebyshev polynomials.
"""

import torch
import torch.nn as nn
import numpy as np
from .cheb_kan_layer import ChebKANLayer


class InputNormalizer(nn.Module):
    """
    Normalizes inputs to [-1, 1] range for Chebyshev domain.
    Uses running statistics (like BatchNorm) or fixed bounds.
    """
    def __init__(self, n_features: int, mode='adaptive'):
        super().__init__()
        self.mode = mode
        if mode == 'adaptive':
            # Running min/max with momentum-based update
            self.register_buffer('running_min', torch.zeros(n_features))
            self.register_buffer('running_max', torch.ones(n_features))
            self.register_buffer('initialized', torch.tensor(False))
            self.momentum = 0.01
        # 'tanh' mode: just apply tanh (simple, no state)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'tanh':
            return torch.tanh(x)
        
        if self.mode == 'adaptive':
            if self.training:
                batch_min = x.min(dim=0).values
                batch_max = x.max(dim=0).values
                
                if not self.initialized:
                    self.running_min.copy_(batch_min)
                    self.running_max.copy_(batch_max)
                    self.initialized.fill_(True)
                else:
                    self.running_min.lerp_(batch_min, self.momentum)
                    self.running_max.lerp_(batch_max, self.momentum)
            
            # Normalize to [-1, 1]
            range_val = (self.running_max - self.running_min).clamp(min=1e-6)
            x_norm = 2.0 * (x - self.running_min) / range_val - 1.0
            return x_norm.clamp(-1.0, 1.0)
        
        return x


class ChebKAN(nn.Module):
    """
    Multi-layer Chebyshev KAN.
    
    Parameters
    ----------
    width : list[int]
        Width of each layer, e.g. [4, 8, 8, 2] means:
        input=4, hidden1=8, hidden2=8, output=2
    degree : int
        Degree of Chebyshev polynomials for all layers.
    masks : list[torch.Tensor] or None
        Topology masks for each layer. If None, fully connected.
    input_normalize : str
        Input normalization mode: 'tanh', 'adaptive', or 'none'
    """
    
    def __init__(self, width: list, degree: int = 5, masks=None, input_normalize='tanh'):
        super().__init__()
        self.width = width
        self.depth = len(width) - 1
        self.degree = degree
        
        # Input normalization
        if input_normalize != 'none':
            self.normalizer = InputNormalizer(width[0], mode=input_normalize)
        else:
            self.normalizer = None
        
        # Build layers
        self.layers = nn.ModuleList()
        for l in range(self.depth):
            mask = masks[l] if masks is not None else None
            layer = ChebKANLayer(
                in_features=width[l],
                out_features=width[l + 1],
                degree=degree,
                mask=mask
            )
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all layers.
        
        Between layers, we apply tanh normalization to keep values in [-1, 1]
        for the next layer's Chebyshev computation.
        """
        # Normalize input to [-1, 1]
        if self.normalizer is not None:
            x = self.normalizer(x)
        else:
            x = torch.tanh(x)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Inter-layer normalization: keep values in [-1, 1] for next layer
            # Don't apply after last layer (output should be raw logits for RL)
            if i < self.depth - 1:
                x = torch.tanh(x)
        
        return x
    
    def get_all_coefficients(self):
        """Return all Chebyshev coefficients as a flat numpy array (for GA encoding)."""
        all_coeffs = []
        for layer in self.layers:
            all_coeffs.append(layer.coeffs.detach().cpu().numpy().flatten())
        return np.concatenate(all_coeffs)
    
    def set_all_coefficients(self, flat_coeffs: np.ndarray):
        """Set all Chebyshev coefficients from a flat numpy array."""
        offset = 0
        with torch.no_grad():
            for layer in self.layers:
                size = layer.coeffs.numel()
                chunk = flat_coeffs[offset:offset + size]
                layer.coeffs.copy_(
                    torch.tensor(chunk, dtype=layer.coeffs.dtype).reshape(layer.coeffs.shape)
                )
                offset += size
    
    def num_parameters(self):
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def spectral_summary(self):
        """Print summary of spectral energy distribution per layer."""
        summary = []
        for i, layer in enumerate(self.layers):
            coeffs = layer.coeffs.detach()
            # Compute energy per degree
            energy = (coeffs ** 2).sum(dim=(0, 1))  # shape: (degree+1,)
            total_energy = energy.sum().item()
            summary.append({
                'layer': i,
                'shape': f'{layer.in_features}→{layer.out_features}',
                'total_energy': total_energy,
                'energy_per_degree': energy.cpu().numpy(),
            })
        return summary
