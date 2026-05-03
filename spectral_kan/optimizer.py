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


def _compute_novelty_scores(behaviors, archive, k=5):
    """
    Compute novelty score for each individual based on behavioral distance
    to k-nearest neighbors in current population + archive.
    
    Novelty Search (Lehman & Stanley, 2011):
    novelty(i) = mean distance to k-nearest neighbors in behavior space.
    
    Parameters
    ----------
    behaviors : list of numpy arrays (or None)
        Behavior descriptors for current population.
    archive : list of numpy arrays
        Historical behavior archive.
    k : int
        Number of nearest neighbors.
    
    Returns
    -------
    list of float: novelty scores for each individual.
    """
    valid_behaviors = []
    valid_indices = []
    
    for i, b in enumerate(behaviors):
        if b is not None:
            valid_behaviors.append(b)
            valid_indices.append(i)
    
    if not valid_behaviors:
        return [0.0] * len(behaviors)
    
    # Pool = current population behaviors + archive
    pool = valid_behaviors + list(archive) if archive else valid_behaviors
    pool_array = np.array(pool)
    
    novelty_scores = [0.0] * len(behaviors)
    
    for idx, b in zip(valid_indices, valid_behaviors):
        # Compute distances to all in pool
        dists = np.linalg.norm(pool_array - b, axis=1)
        # k-nearest (excluding self → start from index 1 after sort)
        sorted_dists = np.sort(dists)
        knn_dists = sorted_dists[1:k+1] if len(sorted_dists) > k else sorted_dists[1:]
        novelty_scores[idx] = float(np.mean(knn_dists)) if len(knn_dists) > 0 else 0.0
    
    return novelty_scores


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
        elitism_count: int = 2,
        novelty_weight: float = 0.0,
        novelty_k: int = 5,
        archive_prob: float = 0.1,
        archive_max_size: int = 200
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
        
        # Novelty Search parameters
        self.novelty_weight = novelty_weight  # α in: fitness_combined = (1-α)*reward_fitness + α*(-novelty)
        self.novelty_k = novelty_k
        self.archive_prob = archive_prob  # Probability of adding to behavior archive
        self.archive_max_size = archive_max_size
        self.behavior_archive = []  # Stores diverse behavior descriptors
        
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
        Main Lamarckian GA loop with parallel evaluation and Novelty Search.
        
        When novelty_weight > 0, uses combined fitness:
            combined = (1 - α) * reward_fitness + α * (-novelty_score)
        
        This encourages behavioral diversity, helping escape deceptive local optima.
        
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
                behaviors = [res[2] for res in results]
                
                # [Lamarckian] Inherit trained weights
                for ind, w in zip(self.population, weights_list):
                    if w is not None:
                        ind.inherit_weights(w)
                
                # [Novelty Search] Compute combined fitness if enabled
                if self.novelty_weight > 0:
                    novelty_scores = _compute_novelty_scores(
                        behaviors, self.behavior_archive, k=self.novelty_k
                    )
                    
                    # Normalize novelty scores to same scale as fitnesses
                    valid_novelties = [n for n in novelty_scores if n > 0]
                    if valid_novelties:
                        max_novelty = max(valid_novelties)
                        norm_novelties = [n / max_novelty if max_novelty > 0 else 0 for n in novelty_scores]
                    else:
                        norm_novelties = [0.0] * len(novelty_scores)
                    
                    # Combined fitness: lower is better, novelty is reward (subtract)
                    valid_fits = [f for f in fitnesses if f != float('inf')]
                    fit_range = max(valid_fits) - min(valid_fits) if len(valid_fits) > 1 else 1.0
                    
                    selection_fitnesses = []
                    for f, n in zip(fitnesses, norm_novelties):
                        if f == float('inf'):
                            selection_fitnesses.append(float('inf'))
                        else:
                            # Subtract novelty (higher novelty → lower combined fitness → better)
                            combined = (1 - self.novelty_weight) * f - self.novelty_weight * n * abs(fit_range)
                            selection_fitnesses.append(combined)
                    
                    # Update behavior archive (random subset)
                    for b in behaviors:
                        if b is not None and np.random.random() < self.archive_prob:
                            self.behavior_archive.append(b)
                    # Keep archive bounded
                    if len(self.behavior_archive) > self.archive_max_size:
                        self.behavior_archive = self.behavior_archive[-self.archive_max_size:]
                else:
                    selection_fitnesses = fitnesses
                
                # Update best (always based on TRUE fitness, not novelty-adjusted)
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
                if self.novelty_weight > 0:
                    gen_stats['avg_novelty'] = float(np.mean([n for n in novelty_scores if n > 0])) if any(n > 0 for n in novelty_scores) else 0.0
                    gen_stats['archive_size'] = len(self.behavior_archive)
                self.history.append(gen_stats)
                
                pbar.set_postfix({
                    'Best': f'{-self.best_fitness:.1f}',
                    'Avg': f'{-avg_fit:.1f}',
                    'Valid': f'{len(valid_fitnesses)}/{self.pop_size}'
                })
                
                if logger:
                    novelty_info = ""
                    if self.novelty_weight > 0:
                        novelty_info = f" | Novelty: {gen_stats.get('avg_novelty', 0):.3f} | Archive: {len(self.behavior_archive)}"
                    logger.info(
                        f"Gen {gen+1}/{self.max_gen} | "
                        f"Best Reward: {-self.best_fitness:.2f} | "
                        f"Avg Reward: {-avg_fit:.2f} | "
                        f"Valid: {len(valid_fitnesses)}/{self.pop_size}"
                        f"{novelty_info}"
                    )
                
                # Generate next generation (use selection_fitnesses which may include novelty)
                new_population = self._next_generation(selection_fitnesses)
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
