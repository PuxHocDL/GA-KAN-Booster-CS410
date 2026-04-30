import numpy as np

class ChromosomeConfig:
    def __init__(self, n, m, d_max, u_max, b_grid_len=6, b_depth_len=2):
        self.n = n
        self.m = m
        self.d_max = d_max
        self.u_max = u_max
        self.b_grid_len = b_grid_len
        self.b_depth_len = b_depth_len
        
        # Determine max architecture width
        # Layer 0 (input): n
        # Layer 1 to d_max-1 (hidden): u_max
        # Layer d_max (output): m
        self.max_width = [self.n] + [self.u_max] * (self.d_max - 1) + [self.m]
        
        # Precompute topology block sizes and offsets
        self.topo_blocks = []
        for i in range(self.d_max):
            size = self.max_width[i] * self.max_width[i+1]
            self.topo_blocks.append(size)
            
        self.b_topo = sum(self.topo_blocks)
        self.b_total = self.b_depth_len + self.b_grid_len + self.b_topo
        
class Chromosome:
    def __init__(self, config: ChromosomeConfig, bits=None):
        self.config = config
        if bits is None:
            self.bits = np.random.randint(2, size=config.b_total)
        else:
            self.bits = np.array(bits)
            
    def decode(self):
        """
        Decodes the chromosome into depth, grid, and topology masks.
        """
        idx = 0
        
        # 1. Decode depth
        depth_bits = self.bits[idx : idx + self.config.b_depth_len]
        idx += self.config.b_depth_len
        # binary array to decimal
        depth_val = depth_bits.dot(1 << np.arange(depth_bits.size)[::-1])
        target_depth = int(depth_val) + 1  # 1 to 2^b_depth_len
        # constrain to d_max
        target_depth = min(target_depth, self.config.d_max)
        
        # 2. Decode grid
        grid_bits = self.bits[idx : idx + self.config.b_grid_len]
        idx += self.config.b_grid_len
        grid_val = grid_bits.dot(1 << np.arange(grid_bits.size)[::-1])
        grid_value = int(grid_val) + 1  # 1 to 2^b_grid_len
        
        # 3. Decode topology
        masks = []
        
        # We need to extract masks for layer 0 to target_depth
        # The structure of sub-network: [n, u_max, ..., m] (length target_depth + 1)
        # Layer 0 -> Layer 1
        # Layer 1 -> Layer 2
        # ...
        # Layer (target_depth-1) -> Layer (target_depth)
        
        topo_idx = idx
        
        for i in range(self.config.d_max):
            block_size = self.config.topo_blocks[i]
            block_bits = self.bits[topo_idx : topo_idx + block_size]
            topo_idx += block_size
            
            # Map the bits to 2D mask matrix: shape (in_features, out_features)
            in_features = self.config.max_width[i]
            out_features = self.config.max_width[i+1]
            
            mask = block_bits.reshape((in_features, out_features))
            masks.append(mask)
            
        # Degradation & Zero-mask
        # For target_depth < d_max, we use:
        # masks[0], masks[1], ..., masks[target_depth-2]
        # and for the final connection to output, we use the LAST mask (masks[-1])
        
        active_masks = []
        if target_depth == 1:
            # Special case: direct connection from n to m
            # We don't have an n x m block in the general case unless d_max=1
            # Wait, if target_depth=1, we need an n x m mask. 
            # PyKAN depth 1: KAN(width=[n, m]).
            # Where do we get n x m bits?
            # We can just take the first n * m bits from masks[0] if it's large enough,
            # or just take the first n*m bits from the entire topology.
            # To be safe, let's take the first n * m bits of the topology string.
            flat_topo = self.bits[idx:]
            req_size = self.config.n * self.config.m
            req_bits = flat_topo[:req_size]
            active_masks.append(req_bits.reshape((self.config.n, self.config.m)))
        else:
            for i in range(target_depth - 1):
                active_masks.append(masks[i])
            # The last layer connects u_max to m
            # We use masks[-1] which has shape (u_max, m)
            active_masks.append(masks[-1])
            
        return target_depth, grid_value, active_masks

def is_valid_topology(active_masks):
    """
    Checks if there is a path from at least one input node to at least one output node.
    active_masks: list of 2D numpy arrays of shape (in_nodes, out_nodes)
    """
    if not active_masks:
        return False
        
    n_inputs = active_masks[0].shape[0]
    n_outputs = active_masks[-1].shape[1]
    
    # Active nodes in current layer
    # Initially all input nodes are active
    current_active_nodes = np.ones(n_inputs, dtype=bool)
    
    for mask in active_masks:
        # mask is (in_nodes, out_nodes)
        # We want to find which out_nodes can be reached
        # A node j in out_nodes is reachable if there is some i in in_nodes
        # where current_active_nodes[i] is True AND mask[i, j] == 1
        
        # Matrix multiplication with boolean
        next_active_nodes = np.dot(current_active_nodes, mask) > 0
        current_active_nodes = next_active_nodes
        
        if not np.any(current_active_nodes):
            return False # Path disconnected early
            
    # Finally check if any output node is reachable
    return bool(np.any(current_active_nodes))
