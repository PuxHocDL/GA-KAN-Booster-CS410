"""
Visualization for Spectral (Chebyshev) KAN.

Generates KAN-style architecture plots showing:
- Nodes (input, hidden, output) 
- Edges with activation function shapes (Chebyshev polynomial curves)
- Edge thickness based on coefficient magnitude
- Topology mask (active vs inactive edges)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import torch
import os


def plot_cheb_kan(model, title="ChebKAN Policy", save_path=None, figsize=None):
    """
    Plot the ChebKAN architecture with activation function shapes on edges.
    
    Similar to PyKAN's .plot() but for Chebyshev polynomial activations.
    
    Parameters
    ----------
    model : ChebKAN
        The model to visualize.
    title : str
        Plot title.
    save_path : str or None
        If provided, save the figure to this path.
    figsize : tuple or None
        Figure size. If None, auto-computed based on network dimensions.
    """
    width = model.width
    depth = model.depth
    
    if figsize is None:
        max_width = max(width)
        figsize = (3 * max_width, 3 * (depth + 1))
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(-0.5, max(width) - 0.5)
    ax.set_ylim(-0.5, depth + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Node positions: layer l has width[l] nodes centered
    node_positions = []  # node_positions[l] = list of (x, y) for each node in layer l
    for l in range(depth + 1):
        n_nodes = width[l]
        max_w = max(width)
        x_offset = (max_w - n_nodes) / 2.0
        y = depth - l  # input at bottom, output at top
        positions = [(x_offset + i, y) for i in range(n_nodes)]
        node_positions.append(positions)
    
    # Draw edges with activation function mini-plots
    for l, layer in enumerate(model.layers):
        coeffs = layer.coeffs.detach().cpu().numpy()  # (in, out, degree+1)
        mask = layer.mask.detach().cpu().numpy()  # (in, out)
        in_f, out_f = layer.in_features, layer.out_features
        
        # Compute edge strengths (L2 norm of coefficients)
        edge_strength = np.sqrt((coeffs ** 2).sum(axis=-1))  # (in, out)
        max_strength = edge_strength.max() + 1e-8
        
        for i in range(in_f):
            for j in range(out_f):
                if mask[i, j] < 0.5:
                    continue  # Edge pruned by topology mask
                
                x1, y1 = node_positions[l][i]
                x2, y2 = node_positions[l + 1][j]
                
                # Edge properties
                strength = edge_strength[i, j] / max_strength
                alpha = 0.2 + 0.8 * strength
                linewidth = 0.5 + 2.5 * strength
                
                # Draw edge line
                ax.plot([x1, x2], [y1, y2], 
                       color='gray', alpha=alpha, linewidth=linewidth,
                       zorder=1)
                
                # Draw mini activation function at midpoint
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                if strength > 0.3:  # Only show strong activations
                    _draw_mini_activation(ax, mid_x, mid_y, coeffs[i, j], 
                                         size=0.15, alpha=alpha)
    
    # Draw nodes
    for l in range(depth + 1):
        for i, (x, y) in enumerate(node_positions[l]):
            if l == 0:
                color = 'black'
                size = 0.12
            elif l == depth:
                color = 'black'
                size = 0.12
            else:
                color = 'white'
                size = 0.15
            
            circle = plt.Circle((x, y), size, color=color, 
                              ec='black', linewidth=1.5, zorder=3)
            ax.add_patch(circle)
            
            # Add ⊕ symbol for hidden/output nodes (summation)
            if l > 0:
                ax.text(x, y, '⊕', ha='center', va='center', 
                       fontsize=8, fontweight='bold', zorder=4)
    
    # Add layer labels
    for l in range(depth + 1):
        max_w = max(width)
        if l == 0:
            label = f"Input ({width[l]})"
        elif l == depth:
            label = f"Output ({width[l]})"
        else:
            label = f"Hidden ({width[l]})"
        y = depth - l
        ax.text(max_w + 0.3, y, label, fontsize=9, va='center', color='gray')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Architecture plot saved to {save_path}")
    
    plt.close(fig)
    return fig


def _draw_mini_activation(ax, cx, cy, coeffs, size=0.15, alpha=1.0):
    """Draw a tiny Chebyshev activation curve at position (cx, cy)."""
    # Generate curve points
    x = np.linspace(-1, 1, 30)
    y = np.zeros_like(x)
    
    # Evaluate Chebyshev polynomial: φ(x) = Σ c_i T_i(x)
    T_prev = np.ones_like(x)  # T_0
    T_curr = x.copy()  # T_1
    
    y += coeffs[0] * T_prev
    if len(coeffs) > 1:
        y += coeffs[1] * T_curr
    
    for d in range(2, len(coeffs)):
        T_next = 2 * x * T_curr - T_prev
        y += coeffs[d] * T_next
        T_prev = T_curr
        T_curr = T_next
    
    # Normalize to fit in mini-box
    y_range = y.max() - y.min()
    if y_range > 1e-6:
        y_norm = (y - y.min()) / y_range - 0.5  # center around 0
    else:
        y_norm = np.zeros_like(y)
    
    # Scale and position
    plot_x = cx + x * size * 0.8
    plot_y = cy + y_norm * size * 1.5
    
    # Draw background box
    rect = FancyBboxPatch((cx - size, cy - size * 0.8), 2 * size, 1.6 * size,
                          boxstyle="round,pad=0.02",
                          facecolor='white', edgecolor='lightgray',
                          alpha=0.7 * alpha, linewidth=0.5, zorder=2)
    ax.add_patch(rect)
    
    # Draw curve
    ax.plot(plot_x, plot_y, color='blue', linewidth=0.8, alpha=alpha, zorder=2.5)


def plot_spectral_energy(model, title="Spectral Energy Distribution", save_path=None):
    """
    Plot the spectral energy distribution across Chebyshev degrees for each layer.
    Shows how much energy (importance) each frequency band has.
    """
    summary = model.spectral_summary()
    n_layers = len(summary)
    
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 3))
    if n_layers == 1:
        axes = [axes]
    
    for idx, (ax, layer_info) in enumerate(zip(axes, summary)):
        energy = layer_info['energy_per_degree']
        degrees = np.arange(len(energy))
        
        # Color bars: low-freq = blue, high-freq = red
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(energy)))
        
        ax.bar(degrees, energy, color=colors, edgecolor='gray', linewidth=0.5)
        ax.set_xlabel('Chebyshev Degree')
        ax.set_ylabel('Energy (Σ c²)')
        ax.set_title(f"Layer {idx} ({layer_info['shape']})")
        ax.set_xticks(degrees)
    
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Spectral energy plot saved to {save_path}")
    
    plt.close(fig)
    return fig
