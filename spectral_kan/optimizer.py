"""
Spectral GA Optimizer: Lamarckian Evolution for Chebyshev KAN.

Uses ProcessPoolExecutor for parallel evaluation across CPU cores.
Supports frequency-domain crossover and mutation operators.
"""

import copy
import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import concurrent.futures

from .chromosome import SpectralChromosome, SpectralConfig
from .genetic_operators import SelectionStrategy, CrossoverStrategy, MutationStrategy
from .fitness_rl import evaluate_spectral_fitness_rl


def _worker_eval(args):
    """Process-pool worker: evaluate a single SpectralChromosome."""
    import torch
    torch.set_num_threads(1)
    individual, env_name, n_steps, device, n_train_iterations, n_train_iterations_elite, max_steps = args
    return evaluate_spectral_fitness_rl(
        individual, env_name, n_steps, device,
        envs=None,
        vectorization_mode='sync',
        n_train_iterations=n_train_iterations,
        n_train_iterations_elite=n_train_iterations_elite,
        max_steps=max_steps
    )


class SpectralGAOptimizer:
    """
    GA optimizer for Spectral (Chebyshev) KAN with Lamarckian Evolution.
    
    Key differences from B-spline GA-KAN:
    - No grid parameter to evolve (replaced by Chebyshev degree)
    - Frequency-domain aware crossover and mutation on weights
    - More stable weight inheritance (no grid updates)
    - Dense topology initialization
    """
    
    def __init__(
        self,
        config: SpectralConfig,
        selection_strategy: SelectionStrategy,
        crossover_strategy: CrossoverStrategy,
        mutation_strategy: MutationStrategy,
        pop_size: int = 20,
        max_gen: int = 15,
        N_steps: int = 8,
        device: str = 'cpu',
        n_train_iterations: int = 5,
        n_train_iterations_elite: int = 15,
        max_steps_per_iter: int = 500,
        vectorization_mode: str = 'sync',
        dense_init: bool = True,
        num_workers: int = None,
        elitism_count: int = 2
    ):
        self.config = config
        self.selection_strategy = selection_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy
        
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.N_steps = N_steps
        self.device = device
        self.n_train_iterations = n_train_iterations
        self.n_train_iterations_elite = n_train_iterations_elite
        self.max_steps_per_iter = max_steps_per_iter
        self.vectorization_mode = vectorization_mode
        self.dense_init = dense_init
        self.elitism_count = elitism_count
        
        # Worker count
        max_safe_workers = max(1, (os.cpu_count() or 4) // max(1, N_steps // 2))
        self.num_workers = min(num_workers or pop_size, max_safe_workers, 60)
        
        self.population = []
        self.best_individual = None
        self.best_fitness = float('inf')
        self.history = []  # Track generational stats
    
    def initialize_population(self):
        """Initialize population with dense topology and favorable architecture."""
        population = []
        for _ in range(self.pop_size):
            ind = SpectralChromosome(self.config)
            if self.dense_init:
                # Bias topology bits toward 1 (85% active) for denser networks
                arch_len = (self.config.b_depth_len + self.config.b_degree_len + 
                           self.config.b_width_len)
                topo_bits = ind.bits[arch_len:]
                ind.bits[arch_len:] = (np.random.random(len(topo_bits)) < 0.85).astype(int)
            population.append(ind)
        self.population = population
    
    def run(self, env_name: str, logger=None):
        """
        Main Lamarckian GA loop with parallel evaluation.
        
        Returns
        -------
        best_individual, best_fitness
        """
        self.initialize_population()
        
        ctx = mp.get_context('spawn')
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=ctx
        )
        
        pbar = tqdm(range(self.max_gen), desc="Spectral GA Optimization", unit="gen")
        try:
            for gen in pbar:
                # Parallel fitness evaluation
                payload = [
                    (ind, env_name, self.N_steps, 'cpu',
                     self.n_train_iterations, self.n_train_iterations_elite,
                     self.max_steps_per_iter)
                    for ind in self.population
                ]
                results = list(executor.map(_worker_eval, payload))
                
                fitnesses = [res[0] for res in results]
                weights_list = [res[1] for res in results]
                
                # [Lamarckian] Inherit trained weights
                for ind, w in zip(self.population, weights_list):
                    if w is not None:
                        ind.inherit_weights(w)
                
                # Update best
                for ind, fit in zip(self.population, fitnesses):
                    if fit < self.best_fitness:
                        self.best_fitness = fit
                        self.best_individual = copy.deepcopy(ind)
                
                # Log metrics
                valid_fitnesses = [f for f in fitnesses if f != float('inf')]
                avg_fit = sum(valid_fitnesses) / len(valid_fitnesses) if valid_fitnesses else float('inf')
                
                gen_stats = {
                    'gen': gen + 1,
                    'best_reward': -self.best_fitness,
                    'avg_reward': -avg_fit,
                    'pop_valid': len(valid_fitnesses),
                }
                self.history.append(gen_stats)
                
                pbar.set_postfix({
                    'Best': f'{-self.best_fitness:.1f}',
                    'Avg': f'{-avg_fit:.1f}',
                    'Valid': f'{len(valid_fitnesses)}/{self.pop_size}'
                })
                
                if logger:
                    logger.info(
                        f"Gen {gen+1}/{self.max_gen} | "
                        f"Best Reward: {-self.best_fitness:.2f} | "
                        f"Avg Reward: {-avg_fit:.2f} | "
                        f"Valid: {len(valid_fitnesses)}/{self.pop_size}"
                    )
                
                # Generate next generation
                new_population = self._next_generation(fitnesses)
                self.population = new_population
                
        finally:
            executor.shutdown(wait=False)
        
        return self.best_individual, self.best_fitness
    
    def _next_generation(self, fitnesses):
        """Create next generation with elitism + crossover + mutation."""
        new_population = []
        
        # Elitism: keep top-k individuals with their weights
        sorted_indices = np.argsort(fitnesses)
        for i in range(min(self.elitism_count, len(sorted_indices))):
            idx = sorted_indices[i]
            if fitnesses[idx] != float('inf'):
                elite = copy.deepcopy(self.population[idx])
                new_population.append(elite)
        
        # Always keep the global best
        if self.best_individual is not None and len(new_population) == 0:
            new_population.append(copy.deepcopy(self.best_individual))
        
        # Fill remaining with crossover + mutation
        while len(new_population) < self.pop_size:
            parents = self.selection_strategy.select(self.population, fitnesses, num_parents=2)
            p1, p2 = parents[0], parents[1]
            
            c1, c2 = self.crossover_strategy.crossover(p1, p2)
            
            self.mutation_strategy.mutate(c1)
            self.mutation_strategy.mutate(c2)
            
            new_population.append(c1)
            if len(new_population) < self.pop_size:
                new_population.append(c2)
        
        return new_population
