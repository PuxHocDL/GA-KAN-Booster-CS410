"""
ChebKANLayer: Kolmogorov-Arnold Network layer using Chebyshev polynomials.

Instead of B-splines with local grid support, each edge activation is:
    φ(x) = Σ_{i=0}^{D} c_i * T_i(x)

where T_i(x) are Chebyshev polynomials of the first kind, computed via recurrence:
    T_0(x) = 1
    T_1(x) = x
    T_{n+1}(x) = 2x * T_n(x) - T_{n-1}(x)

Advantages over B-splines:
- Global support: no out-of-grid issues during RL exploration
- No grid updates needed: weights are stable for Lamarckian inheritance
- Frequency-domain interpretation: low-order = coarse shape, high-order = fine details
- Fast recurrence computation: O(D) per input element
"""

import torch
import torch.nn as nn
import math


class ChebKANLayer(nn.Module):
    """
    A single KAN layer where each edge (i→j) has its own Chebyshev polynomial activation.
    
    Parameters
    ----------
    in_features : int
        Number of input nodes.
    out_features : int
        Number of output nodes.
    degree : int
        Maximum degree of Chebyshev polynomials (number of coefficients = degree + 1).
    mask : torch.Tensor or None
        Binary mask of shape (in_features, out_features) for topology control.
        If None, all edges are active.
    """
    
    def __init__(self, in_features: int, out_features: int, degree: int = 5, mask=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        # Chebyshev coefficients: shape (in_features, out_features, degree+1)
        # Initialize with Xavier-like scaling adapted for polynomial basis
        self.coeffs = nn.Parameter(
            torch.empty(in_features, out_features, degree + 1)
        )
        self._init_coeffs()
        
        # Topology mask (not a parameter - controlled by GA)
        if mask is not None:
            self.register_buffer('mask', mask.float())
        else:
            self.register_buffer('mask', torch.ones(in_features, out_features))
    
    def _init_coeffs(self):
        """Initialize Chebyshev coefficients with decaying magnitudes for higher orders."""
        # Lower-order coefficients get larger init (they define coarse shape)
        # Higher-order get smaller init (fine details)
        with torch.no_grad():
            fan_in = self.in_features
            std = 1.0 / math.sqrt(fan_in)
            for d in range(self.degree + 1):
                # Decay factor: higher degree → smaller initialization
                decay = 1.0 / (1.0 + d)
                self.coeffs[:, :, d].normal_(0, std * decay)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, in_features). 
            Should be normalized to [-1, 1] for Chebyshev domain.
        
        Returns
        -------
        torch.Tensor
            Output of shape (batch_size, out_features).
        """
        # Clamp to [-1, 1] to stay in Chebyshev domain (safety)
        x = torch.clamp(x, -1.0, 1.0)
        
        batch_size = x.shape[0]
        
        # Compute Chebyshev polynomials T_0(x), T_1(x), ..., T_D(x)
        # x shape: (batch, in_features)
        # We need T_d(x) for each input feature: shape (batch, in_features, degree+1)
        cheb = self._compute_chebyshev(x)  # (batch, in_features, degree+1)
        
        # Compute activations: φ_{i,j}(x_i) = Σ_d c_{i,j,d} * T_d(x_i)
        # cheb: (batch, in_features, degree+1)
        # coeffs: (in_features, out_features, degree+1)
        # We want: output_j = Σ_i φ_{i,j}(x_i) = Σ_i Σ_d c_{i,j,d} * T_d(x_i)
        
        # Efficient: einsum
        # result[b, j] = Σ_i Σ_d cheb[b, i, d] * coeffs[i, j, d] * mask[i, j]
        # First compute per-edge activation: (batch, in_features, out_features)
        # = Σ_d cheb[b,i,d] * coeffs[i,j,d]
        activations = torch.einsum('bid,ijd->bij', cheb, self.coeffs)
        
        # Apply topology mask
        activations = activations * self.mask.unsqueeze(0)  # (batch, in, out)
        
        # Sum over input dimension
        output = activations.sum(dim=1)  # (batch, out_features)
        
        return output
    
    def _compute_chebyshev(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Chebyshev polynomials T_0(x) through T_D(x) using recurrence.
        
        Parameters
        ----------
        x : torch.Tensor of shape (batch, in_features)
        
        Returns
        -------
        torch.Tensor of shape (batch, in_features, degree+1)
        """
        # T_0 = 1, T_1 = x
        cheb_list = [torch.ones_like(x)]  # T_0
        if self.degree >= 1:
            cheb_list.append(x)  # T_1
        
        # Recurrence: T_{n+1} = 2x * T_n - T_{n-1}
        for n in range(1, self.degree):
            t_next = 2.0 * x * cheb_list[n] - cheb_list[n - 1]
            cheb_list.append(t_next)
        
        # Stack: (batch, in_features, degree+1)
        return torch.stack(cheb_list, dim=-1)
    
    def get_coefficients(self):
        """Return the Chebyshev coefficients as numpy array."""
        return self.coeffs.detach().cpu().numpy()
    
    def low_freq_coeffs(self, cutoff: int = None):
        """Return low-frequency (low-order) coefficients."""
        if cutoff is None:
            cutoff = max(1, self.degree // 2)
        return self.coeffs[:, :, :cutoff]
    
    def high_freq_coeffs(self, cutoff: int = None):
        """Return high-frequency (high-order) coefficients."""
        if cutoff is None:
            cutoff = max(1, self.degree // 2)
        return self.coeffs[:, :, cutoff:]
