"""
Spectral Chromosome: Genetic encoding for Chebyshev KAN architecture.

Encodes:
- Network depth (number of layers)
- Chebyshev degree (polynomial order)
- Hidden layer width
- Topology masks (which edges are active)

Unlike B-spline KAN, there's no grid parameter. Instead we control the
Chebyshev degree which determines expressiveness.
"""

import numpy as np
import copy


class SpectralConfig:
    """
    Configuration for Spectral KAN chromosome encoding.
    
    Parameters
    ----------
    n : int
        Input dimension.
    m : int
        Output dimension.
    d_max : int
        Maximum depth (number of layers).
    u_max : int
        Maximum hidden layer width.
    degree_max : int
        Maximum Chebyshev polynomial degree.
    b_depth_len : int
        Bits for encoding depth.
    b_degree_len : int
        Bits for encoding Chebyshev degree.
    b_width_len : int
        Bits for encoding hidden width.
    """
    
    def __init__(self, n, m, d_max=3, u_max=16, degree_max=10,
                 b_depth_len=2, b_degree_len=4, b_width_len=4):
        self.n = n
        self.m = m
        self.d_max = d_max
        self.u_max = u_max
        self.degree_max = degree_max
        self.b_depth_len = b_depth_len
        self.b_degree_len = b_degree_len
        self.b_width_len = b_width_len
        
        # Max width layout: [n, u_max, ..., u_max, m]
        self.max_width = [self.n] + [self.u_max] * (self.d_max - 1) + [self.m]
        
        # Topology block sizes
        self.topo_blocks = []
        for i in range(self.d_max):
            size = self.max_width[i] * self.max_width[i + 1]
            self.topo_blocks.append(size)
        
        self.b_topo = sum(self.topo_blocks)
        self.b_total = self.b_depth_len + self.b_degree_len + self.b_width_len + self.b_topo


class SpectralChromosome:
    """
    Chromosome for Spectral (Chebyshev) KAN.
    
    Binary encoding:
    [depth_bits | degree_bits | width_bits | topology_bits]
    
    Also stores trained weights (Chebyshev coefficients) for Lamarckian inheritance.
    """
    
    def __init__(self, config: SpectralConfig, bits=None, weights=None):
        self.config = config
        if bits is None:
            self.bits = np.random.randint(2, size=config.b_total)
        else:
            self.bits = np.array(bits)
        
        # Lamarckian: store trained model weights
        self.weights = copy.deepcopy(weights) if weights is not None else None
    
    def inherit_weights(self, weights):
        """Save trained weights for Lamarckian inheritance."""
        self.weights = copy.deepcopy(weights)
    
    def has_weights(self):
        return self.weights is not None
    
    def decode(self):
        """
        Decode chromosome into architecture parameters.
        
        Returns
        -------
        depth : int
            Number of layers.
        degree : int
            Chebyshev polynomial degree.
        width : list[int]
            Layer widths [input, hidden..., output].
        active_masks : list[np.ndarray]
            Topology masks for each layer connection.
        """
        idx = 0
        
        # 1. Decode depth (1 to d_max)
        depth_bits = self.bits[idx:idx + self.config.b_depth_len]
        idx += self.config.b_depth_len
        depth_val = depth_bits.dot(1 << np.arange(depth_bits.size)[::-1])
        target_depth = int(depth_val) + 1
        target_depth = min(target_depth, self.config.d_max)
        
        # 2. Decode Chebyshev degree (2 to degree_max)
        degree_bits = self.bits[idx:idx + self.config.b_degree_len]
        idx += self.config.b_degree_len
        degree_val = degree_bits.dot(1 << np.arange(degree_bits.size)[::-1])
        degree = int(degree_val) + 2  # minimum degree 2
        degree = min(degree, self.config.degree_max)
        
        # 3. Decode hidden width (4 to u_max)
        width_bits = self.bits[idx:idx + self.config.b_width_len]
        idx += self.config.b_width_len
        width_val = width_bits.dot(1 << np.arange(width_bits.size)[::-1])
        hidden_width = int(width_val) + 4  # minimum width 4
        hidden_width = min(hidden_width, self.config.u_max)
        
        # Build actual width
        if target_depth == 1:
            width = [self.config.n, self.config.m]
        else:
            width = [self.config.n] + [hidden_width] * (target_depth - 1) + [self.config.m]
        
        # 4. Decode topology masks
        topo_start = self.config.b_depth_len + self.config.b_degree_len + self.config.b_width_len
        topo_bits = self.bits[topo_start:]
        
        active_masks = []
        topo_idx = 0
        
        if target_depth == 1:
            # Direct n→m connection
            req_size = self.config.n * self.config.m
            mask_bits = topo_bits[:req_size]
            if len(mask_bits) < req_size:
                mask_bits = np.pad(mask_bits, (0, req_size - len(mask_bits)), constant_values=1)
            active_masks.append(mask_bits.reshape(self.config.n, self.config.m))
        else:
            for i in range(target_depth):
                if i == 0:
                    in_f, out_f = self.config.n, hidden_width
                elif i == target_depth - 1:
                    in_f, out_f = hidden_width, self.config.m
                else:
                    in_f, out_f = hidden_width, hidden_width
                
                # Extract bits from the full topology block
                block_size = self.config.topo_blocks[min(i, self.config.d_max - 1)]
                block_bits = topo_bits[topo_idx:topo_idx + block_size]
                topo_idx += block_size
                
                # Reshape to actual needed size (may need to take subset)
                req_size = in_f * out_f
                if len(block_bits) >= req_size:
                    mask = block_bits[:req_size].reshape(in_f, out_f)
                else:
                    # Pad with 1s (active) if not enough bits
                    padded = np.ones(req_size, dtype=int)
                    padded[:len(block_bits)] = block_bits[:req_size]
                    mask = padded.reshape(in_f, out_f)
                
                active_masks.append(mask)
        
        return target_depth, degree, width, active_masks
    
    def architecture_string(self):
        """Human-readable architecture description."""
        depth, degree, width, masks = self.decode()
        return f"ChebKAN(width={width}, degree={degree}, params~{self._estimate_params(width, degree)})"
    
    def _estimate_params(self, width, degree):
        """Estimate number of parameters."""
        total = 0
        for i in range(len(width) - 1):
            total += width[i] * width[i + 1] * (degree + 1)
        return total


def is_valid_spectral_topology(active_masks):
    """
    Check if there's at least one active path from input to output.
    """
    if not active_masks:
        return False
    
    n_inputs = active_masks[0].shape[0]
    active_nodes = np.ones(n_inputs, dtype=bool)
    
    for mask in active_masks:
        # Which output nodes can be reached from currently active input nodes?
        reachable = (mask[active_nodes] > 0).any(axis=0)
        if not reachable.any():
            return False
        active_nodes = reachable
    
    return True
