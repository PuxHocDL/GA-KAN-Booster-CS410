"""
Frequency-Domain Genetic Operators for Spectral KAN.

Key insight: Chebyshev coefficients have a natural frequency interpretation.
- Low-order coefficients (c_0, c_1, ..., c_k) → coarse policy shape
- High-order coefficients (c_{k+1}, ..., c_D) → fine-grained details

This enables:
- Low-frequency Crossover: combine coarse policies from two parents
- High-frequency Mutation: add exploration noise only to fine details
- Spectral Energy Mutation: scale coefficient magnitudes by frequency band
"""

import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List, Tuple
from .chromosome import SpectralChromosome, SpectralConfig


# ===============================
# Selection Strategy
# ===============================

class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, population, fitnesses, num_parents=2):
        pass


class TournamentSelection(SelectionStrategy):
    def __init__(self, tournament_size=3):
        self.tournament_size = tournament_size
    
    def select(self, population, fitnesses, num_parents=2):
        parents = []
        for _ in range(num_parents):
            indices = np.random.choice(len(population), size=self.tournament_size, replace=False)
            best_idx = min(indices, key=lambda idx: fitnesses[idx])
            parents.append(population[best_idx])
        return parents


# ===============================
# Crossover Strategies
# ===============================

class CrossoverStrategy(ABC):
    @abstractmethod
    def crossover(self, parent1: SpectralChromosome, parent2: SpectralChromosome) -> Tuple[SpectralChromosome, SpectralChromosome]:
        pass


class SpectralCrossover(CrossoverStrategy):
    """
    Frequency-domain aware crossover.
    
    Strategy:
    1. Architecture bits (depth, degree, width): single-point crossover
    2. Topology bits: uniform crossover with rate pc
    3. Weights (if both parents have them): low-frequency blending
       - Low-freq coefficients: weighted average (blend policies)
       - High-freq coefficients: randomly inherit from one parent
    """
    
    def __init__(self, pc_topology=0.3, freq_cutoff_ratio=0.5, alpha=0.5):
        """
        Parameters
        ----------
        pc_topology : float
            Probability of swapping each topology bit.
        freq_cutoff_ratio : float
            Fraction of coefficients considered "low frequency" (0.5 = first half).
        alpha : float
            Blending factor for low-freq crossover (0.5 = equal blend).
        """
        self.pc_topology = pc_topology
        self.freq_cutoff_ratio = freq_cutoff_ratio
        self.alpha = alpha
    
    def crossover(self, parent1: SpectralChromosome, parent2: SpectralChromosome) -> Tuple[SpectralChromosome, SpectralChromosome]:
        config = parent1.config
        c1_bits = np.copy(parent1.bits)
        c2_bits = np.copy(parent2.bits)
        
        # 1. Architecture crossover (depth + degree + width): single-point
        arch_len = config.b_depth_len + config.b_degree_len + config.b_width_len
        if arch_len > 1:
            crossover_point = random.randint(1, arch_len - 1)
            c1_bits[crossover_point:arch_len] = parent2.bits[crossover_point:arch_len]
            c2_bits[crossover_point:arch_len] = parent1.bits[crossover_point:arch_len]
        
        # 2. Topology crossover: uniform
        topo_start = arch_len
        for i in range(topo_start, len(c1_bits)):
            if random.random() < self.pc_topology:
                c1_bits[i], c2_bits[i] = c2_bits[i], c1_bits[i]
        
        # Create children chromosomes
        child1 = SpectralChromosome(config, bits=c1_bits)
        child2 = SpectralChromosome(config, bits=c2_bits)
        
        # 3. Weight-level crossover (Lamarckian frequency blending)
        if parent1.has_weights() and parent2.has_weights():
            self._frequency_blend_weights(parent1, parent2, child1, child2)
        
        return child1, child2
    
    def _frequency_blend_weights(self, p1, p2, c1, c2):
        """
        Blend weights in frequency domain.
        Low-freq: weighted average → stable policy transfer
        High-freq: random selection from one parent → maintain diversity
        
        Weights are stored as numpy arrays to avoid torch fd-sharing issues.
        """
        w1 = p1.weights
        w2 = p2.weights
        
        if w1 is None or w2 is None:
            return
        
        # Check if architectures are compatible (same keys and shapes)
        if set(w1.keys()) != set(w2.keys()):
            return
        
        for key in w1:
            if w1[key].shape != w2[key].shape:
                return
        
        # Blend weights (numpy-based to stay pickle-safe)
        c1_weights = {}
        c2_weights = {}
        
        for key in w1:
            arr1 = w1[key].astype(np.float32)
            arr2 = w2[key].astype(np.float32)
            
            if 'coeffs' in key and arr1.ndim == 3:
                # This is a Chebyshev coefficient array: (in, out, degree+1)
                degree_plus_1 = arr1.shape[-1]
                cutoff = max(1, int(degree_plus_1 * self.freq_cutoff_ratio))
                
                # Low-frequency: blend
                low1 = arr1[:, :, :cutoff]
                low2 = arr2[:, :, :cutoff]
                blended_low_c1 = self.alpha * low1 + (1 - self.alpha) * low2
                blended_low_c2 = (1 - self.alpha) * low1 + self.alpha * low2
                
                # High-frequency: randomly pick from one parent
                high1 = arr1[:, :, cutoff:]
                high2 = arr2[:, :, cutoff:]
                
                # Random mask for which elements come from which parent
                mask = np.random.random(high1.shape) > 0.5
                high_c1 = np.where(mask, high1, high2)
                high_c2 = np.where(mask, high2, high1)
                
                c1_weights[key] = np.concatenate([blended_low_c1, high_c1], axis=-1)
                c2_weights[key] = np.concatenate([blended_low_c2, high_c2], axis=-1)
            else:
                # Non-coefficient weights: random pick
                if random.random() < 0.5:
                    c1_weights[key] = arr1.copy()
                    c2_weights[key] = arr2.copy()
                else:
                    c1_weights[key] = arr2.copy()
                    c2_weights[key] = arr1.copy()
        
        c1.weights = c1_weights
        c2.weights = c2_weights


# ===============================
# Mutation Strategies
# ===============================

class MutationStrategy(ABC):
    @abstractmethod
    def mutate(self, individual: SpectralChromosome):
        pass


class FrequencyMutation(MutationStrategy):
    """
    Frequency-aware mutation for Spectral KAN.
    
    1. Topology bits: standard bit-flip mutation with rate pm
    2. Architecture bits: lower mutation rate (more disruptive)
    3. Weights (if present): high-frequency noise injection
       - Only mutate high-order Chebyshev coefficients
       - This adds exploration noise without destroying core policy
    """
    
    def __init__(self, pm_topology=0.02, pm_architecture=0.005,
                 weight_mutation_rate=0.3, noise_scale=0.1,
                 freq_cutoff_ratio=0.5):
        """
        Parameters
        ----------
        pm_topology : float
            Bit-flip probability for topology bits.
        pm_architecture : float
            Bit-flip probability for architecture bits (lower = more stable).
        weight_mutation_rate : float
            Probability of mutating weights at all.
        noise_scale : float
            Scale of Gaussian noise added to high-frequency coefficients.
        freq_cutoff_ratio : float
            Fraction below which coefficients are "protected" low-frequency.
        """
        self.pm_topology = pm_topology
        self.pm_architecture = pm_architecture
        self.weight_mutation_rate = weight_mutation_rate
        self.noise_scale = noise_scale
        self.freq_cutoff_ratio = freq_cutoff_ratio
    
    def mutate(self, individual: SpectralChromosome):
        config = individual.config
        arch_len = config.b_depth_len + config.b_degree_len + config.b_width_len
        
        # 1. Architecture bits: low mutation rate
        for i in range(arch_len):
            if random.random() < self.pm_architecture:
                individual.bits[i] = 1 - individual.bits[i]
        
        # 2. Topology bits: standard mutation rate
        for i in range(arch_len, len(individual.bits)):
            if random.random() < self.pm_topology:
                individual.bits[i] = 1 - individual.bits[i]
        
        # 3. Weight mutation: high-frequency noise injection
        if individual.has_weights() and random.random() < self.weight_mutation_rate:
            self._high_freq_noise(individual)
    
    def _high_freq_noise(self, individual: SpectralChromosome):
        """Add Gaussian noise to high-frequency Chebyshev coefficients."""
        weights = individual.weights
        if weights is None:
            return
        
        mutated = {}
        for key, arr in weights.items():
            if 'coeffs' in key and arr.ndim == 3:
                arr = arr.copy()
                degree_plus_1 = arr.shape[-1]
                cutoff = max(1, int(degree_plus_1 * self.freq_cutoff_ratio))
                
                # Only add noise to high-frequency coefficients
                high_freq = arr[:, :, cutoff:]
                noise = np.random.randn(*high_freq.shape).astype(np.float32) * self.noise_scale
                arr[:, :, cutoff:] = high_freq + noise
                mutated[key] = arr
            else:
                mutated[key] = arr.copy()
        
        individual.weights = mutated


class SpectralEnergyMutation(MutationStrategy):
    """
    Alternative mutation: scale the spectral energy of coefficient bands.
    
    Instead of adding noise, this mutation scales entire frequency bands up/down,
    effectively "amplifying" or "dampening" certain frequency components of the policy.
    """
    
    def __init__(self, pm_topology=0.02, pm_architecture=0.005,
                 energy_mutation_rate=0.2, scale_range=(0.8, 1.2)):
        self.pm_topology = pm_topology
        self.pm_architecture = pm_architecture
        self.energy_mutation_rate = energy_mutation_rate
        self.scale_range = scale_range
    
    def mutate(self, individual: SpectralChromosome):
        config = individual.config
        arch_len = config.b_depth_len + config.b_degree_len + config.b_width_len
        
        # Bit mutations
        for i in range(arch_len):
            if random.random() < self.pm_architecture:
                individual.bits[i] = 1 - individual.bits[i]
        
        for i in range(arch_len, len(individual.bits)):
            if random.random() < self.pm_topology:
                individual.bits[i] = 1 - individual.bits[i]
        
        # Energy scaling mutation on weights
        if individual.has_weights() and random.random() < self.energy_mutation_rate:
            self._energy_scale(individual)
    
    def _energy_scale(self, individual: SpectralChromosome):
        """Scale frequency bands of Chebyshev coefficients."""
        weights = individual.weights
        if weights is None:
            return
        
        mutated = {}
        for key, arr in weights.items():
            if 'coeffs' in key and arr.ndim == 3:
                arr = arr.copy()
                degree_plus_1 = arr.shape[-1]
                
                # Randomly scale each degree's coefficients
                for d in range(degree_plus_1):
                    if random.random() < 0.3:  # 30% chance to scale each band
                        scale = random.uniform(*self.scale_range)
                        arr[:, :, d] *= scale
                
                mutated[key] = arr
            else:
                mutated[key] = arr.copy()
        
        individual.weights = mutated
