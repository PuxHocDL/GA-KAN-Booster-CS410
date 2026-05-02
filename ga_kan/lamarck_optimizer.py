import copy
import os
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import multiprocessing as mp
import concurrent.futures

from .chromosome import ChromosomeConfig
from .genetic_operators import SelectionStrategy, CrossoverStrategy, MutationStrategy
from .lamarck_chromosome import LamarckChromosome
from .lamarck_fitness_rl import evaluate_lamarckian_fitness_rl


def _worker_eval(args):
    """Process-pool worker: evaluate a single LamarckChromosome."""
    import torch
    torch.set_num_threads(1)  # Avoid thread oversubscription
    individual, env_name, n_steps, device, n_train_iterations, n_train_iterations_elite, max_steps = args
    return evaluate_lamarckian_fitness_rl(
        individual, env_name, n_steps, device,
        envs=None,  # Each worker creates its own env
        vectorization_mode='sync',
        n_train_iterations=n_train_iterations,
        n_train_iterations_elite=n_train_iterations_elite,
        max_steps=max_steps
    )

class LamarckGAKANOptimizer:
    def __init__(
        self,
        config: ChromosomeConfig,
        selection_strategy: SelectionStrategy,
        crossover_strategy: CrossoverStrategy,
        mutation_strategy: MutationStrategy,
        pop_size: int = 10,
        max_gen: int = 50,
        N_steps: int = 8, # number of parallel environments for vectorized REINFORCE
        device: str = 'cpu',
        n_train_iterations: int = 3,
        n_train_iterations_elite: int = 10,
        max_steps_per_iter: int = 400,
        vectorization_mode: str = 'sync',
        dense_init: bool = True,  # Initialize topology with denser connections
        num_workers: int = None  # Number of parallel processes (None = auto)
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
        # Cap workers: each uses n_envs threads, so limit to avoid oversubscription
        max_safe_workers = max(1, (os.cpu_count() or 4) // max(1, N_steps // 2))
        self.num_workers = min(num_workers or pop_size, max_safe_workers, 60)
        
        self.population = []
        self.best_individual = None
        self.best_fitness = float('inf')
        
    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            ind = LamarckChromosome(self.config)
            if self.dense_init:
                # Bias topology bits toward 1 (80% chance) for denser initial networks
                # This ensures most connections are active initially → stronger models
                # GA will prune connections over time via selection pressure
                topo_start = self.config.b_depth_len + self.config.b_grid_len
                ind.bits[topo_start:] = (np.random.random(self.config.b_topo) < 0.8).astype(int)
            population.append(ind)
        self.population = population
        
    def run(self, env_name: str, logger=None):
        """
        Main Pipeline cho bài toán RL sử dụng Lamarckian Evolution.
        Dùng ProcessPoolExecutor để evaluate song song trên nhiều CPU cores.
        """
        self.initialize_population()
        
        ctx = mp.get_context('spawn')
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=ctx
        )
        
        pbar = tqdm(range(self.max_gen), desc="Lamarckian GA Optimization", unit="gen")
        try:
          for gen in pbar:
            # Parallel evaluation across population using process pool
            payload = [
                (ind, env_name, self.N_steps, 'cpu',
                 self.n_train_iterations, self.n_train_iterations_elite,
                 self.max_steps_per_iter)
                for ind in self.population
            ]
            results = list(executor.map(_worker_eval, payload))
                
            fitnesses = [res[0] for res in results]
            weights_list = [res[1] for res in results]
            
            # [Lamarckian Step] Gắn weights sau khi train ngược lại cho từng cá thể
            for ind, w in zip(self.population, weights_list):
                if w is not None:
                    ind.inherit_weights(w)
                
            # Update best individual
            for ind, fit in zip(self.population, fitnesses):
                if fit < self.best_fitness:
                    self.best_fitness = fit
                    self.best_individual = copy.deepcopy(ind)
            
            # Log metrics
            valid_fitnesses = [f for f in fitnesses if f != float('inf')]
            avg_fit = sum(valid_fitnesses) / len(valid_fitnesses) if valid_fitnesses else float('inf')
            
            log_msg = {'Best Reward': f'{-self.best_fitness:.4f}', 'Avg Reward': f'{-avg_fit:.4f}'}
            pbar.set_postfix(log_msg)
            
            if logger:
                logger.info(f"Generation {gen+1}/{self.max_gen} - Best Reward: {-self.best_fitness:.4f} - Avg Reward: {-avg_fit:.4f}")
            
            # Generate new population
            new_population = []
            
            # Elitism: keep the best individual and its WEIGHTS
            if self.best_individual is not None:
                new_population.append(copy.deepcopy(self.best_individual))
                
            while len(new_population) < self.pop_size:
                # Select 2 parents
                parents = self.selection_strategy.select(self.population, fitnesses, num_parents=2)
                p1, p2 = parents[0], parents[1]
                
                # Crossover
                c1_bits, c2_bits = self.crossover_strategy.crossover(p1, p2)
                
                # Mutation
                self.mutation_strategy.mutate(c1_bits)
                self.mutation_strategy.mutate(c2_bits)
                
                # Convert back to LamarckChromosome
                # Lưu ý: Khi cấu trúc bị thay đổi do Crossover/Mutation, ta khởi tạo weights = None
                # để mô hình bắt đầu train lại từ đầu. Chỉ những cá thể đi qua Elitism mới giữ được weights chuẩn.
                c1 = LamarckChromosome(self.config, bits=c1_bits.bits)
                c2 = LamarckChromosome(self.config, bits=c2_bits.bits)
                
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)
                    
            self.population = new_population
        finally:
            executor.shutdown(wait=False)
            
        return self.best_individual, self.best_fitness
