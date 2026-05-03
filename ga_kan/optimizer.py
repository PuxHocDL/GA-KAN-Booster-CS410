import copy
import os
from typing import Dict
from tqdm import tqdm
import concurrent.futures
import torch
from .chromosome import Chromosome, ChromosomeConfig
from .genetic_operators import SelectionStrategy, CrossoverStrategy, MutationStrategy
from .fitness import evaluate_fitness, build_optimal_model

_PROCESS_D_TRAIN = None
_PROCESS_D_VAL = None

def _set_torch_threads(num_threads: int):
    torch.set_num_threads(max(1, num_threads))

def _numpy_dataset(dataset: Dict):
    return {k: v.detach().cpu().numpy() for k, v in dataset.items()}

def _init_process_worker(num_threads: int, D_train_np: Dict, D_val_np: Dict):
    global _PROCESS_D_TRAIN, _PROCESS_D_VAL
    _set_torch_threads(num_threads)
    _PROCESS_D_TRAIN = {k: torch.from_numpy(v) for k, v in D_train_np.items()}
    _PROCESS_D_VAL = {k: torch.from_numpy(v) for k, v in D_val_np.items()}

def _evaluate_fitness_process(individual, task_type, N_steps, device, use_adam, val_interval):
    if _PROCESS_D_TRAIN is None or _PROCESS_D_VAL is None:
        raise RuntimeError("Process worker dataset was not initialized.")
    return evaluate_fitness(
        individual,
        _PROCESS_D_TRAIN,
        _PROCESS_D_VAL,
        task_type,
        N_steps,
        device,
        use_adam,
        val_interval
    )

class GAKANOptimizer:
    def __init__(
        self,
        config: ChromosomeConfig,
        selection_strategy: SelectionStrategy,
        crossover_strategy: CrossoverStrategy,
        mutation_strategy: MutationStrategy,
        pop_size: int = 20,
        max_gen: int = 50,
        N_steps: int = 20,
        task_type: str = 'regression',
        device: str = 'cpu',
        num_workers: int = None,
        cpu_torch_threads: int = 1,
        parallel_backend: str = 'auto'
    ):
        self.config = config
        self.selection_strategy = selection_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy
        
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.N_steps = N_steps
        self.task_type = task_type
        self.device = device
        self.num_workers = num_workers
        self.cpu_torch_threads = max(1, cpu_torch_threads)
        self.parallel_backend = 'thread' if parallel_backend == 'auto' and os.name == 'nt' else parallel_backend
        if self.parallel_backend == 'auto':
            self.parallel_backend = 'process'
        
        self.population = []
        self.best_individual = None
        self.best_fitness = float('inf')

    def _select_eval_backend(self, D_train: Dict):
        if 'cuda' not in self.device:
            return self.device, 'cpu-parallel', max(1, self.N_steps // 5)

        train_input = D_train['train_input']
        train_size = int(train_input.shape[0]) if train_input.ndim > 0 else 1
        feature_dim = int(train_input.shape[1]) if train_input.ndim > 1 else 1
        workload_hint = train_size * feature_dim * self.pop_size * self.N_steps

        # Small supervised datasets are usually Python-bound here; CPU parallelism
        # tends to beat sequential GPU evaluation despite the requested CUDA device.
        if workload_hint < 2_000_000:
            return 'cpu', 'cpu-parallel', self.N_steps

        return self.device, 'cuda-sequential', max(1, self.N_steps // 2)

    def _resolve_num_workers(self):
        if self.num_workers is not None:
            return max(1, self.num_workers)

        cpu_count = os.cpu_count() or 4
        return max(1, min(self.pop_size, cpu_count))

    def _create_cpu_executor(self, n_workers: int, D_train: Dict, D_val: Dict):
        if self.parallel_backend == 'process':
            D_train_np = _numpy_dataset(D_train)
            D_val_np = _numpy_dataset(D_val)
            return concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_process_worker,
                initargs=(self.cpu_torch_threads, D_train_np, D_val_np)
            )

        return concurrent.futures.ThreadPoolExecutor(max_workers=n_workers)
        
    def initialize_population(self):
        self.population = [Chromosome(self.config) for _ in range(self.pop_size)]
        
    def run(self, D_train: Dict, D_val: Dict):
        """
        Main Pipeline (Vòng lặp GA)
        """
        self.initialize_population()
        
        # Use Adam for GA search phase (much faster), LBFGS only for final model
        use_adam = True
        eval_device, backend_name, val_interval = self._select_eval_backend(D_train)
        if backend_name == 'cpu-parallel' and 'cuda' in self.device:
            print("GA search backend: cpu-parallel (auto-selected for small workload)")
        if backend_name == 'cpu-parallel':
            print(
                f"CPU parallel config: workers={self._resolve_num_workers()}, "
                f"torch_threads_per_worker={self.cpu_torch_threads}, "
                f"backend={self.parallel_backend}"
            )
        elif backend_name == 'cuda-sequential':
            print("GA search backend: cuda-sequential")
        
        executor = None
        prev_threads = None
        try:
            if backend_name == 'cpu-parallel':
                n_workers = self._resolve_num_workers()
                if self.parallel_backend == 'thread':
                    prev_threads = torch.get_num_threads()
                    torch.set_num_threads(self.cpu_torch_threads)
                executor = self._create_cpu_executor(n_workers, D_train, D_val)

            pbar = tqdm(range(self.max_gen), desc="GA Optimization", unit="gen")
            for gen in pbar:
                if backend_name == 'cuda-sequential':
                    fitnesses = [
                        evaluate_fitness(
                            ind, D_train, D_val, self.task_type, self.N_steps,
                            eval_device, use_adam=use_adam, val_interval=val_interval
                        )
                        for ind in self.population
                    ]
                else:
                    if self.parallel_backend == 'process':
                        futures = [
                            executor.submit(
                                _evaluate_fitness_process,
                                ind, self.task_type, self.N_steps,
                                eval_device, use_adam, val_interval
                            )
                            for ind in self.population
                        ]
                    else:
                        futures = [
                            executor.submit(
                                evaluate_fitness,
                                ind, D_train, D_val, self.task_type, self.N_steps,
                                eval_device, use_adam, val_interval
                            )
                            for ind in self.population
                        ]
                    fitnesses = [f.result() for f in futures]

                for ind, fit in zip(self.population, fitnesses):
                    if fit < self.best_fitness:
                        self.best_fitness = fit
                        self.best_individual = copy.deepcopy(ind)

                avg_fit = sum(fitnesses) / len(fitnesses)
                pbar.set_postfix({'Best Loss': f'{self.best_fitness:.4f}', 'Avg Loss': f'{avg_fit:.4f}'})

                # Selection
                # Generate new population Q_t
                new_population = []

                # Elitism: keep the best individual
                if self.best_individual is not None:
                    new_population.append(copy.deepcopy(self.best_individual))

                while len(new_population) < self.pop_size:
                    # Select 2 parents
                    parents = self.selection_strategy.select(self.population, fitnesses, num_parents=2)
                    p1, p2 = parents[0], parents[1]

                    # Crossover
                    c1, c2 = self.crossover_strategy.crossover(p1, p2)

                    # Mutation
                    self.mutation_strategy.mutate(c1)
                    self.mutation_strategy.mutate(c2)

                    new_population.append(c1)
                    if len(new_population) < self.pop_size:
                        new_population.append(c2)

                self.population = new_population
        finally:
            if executor is not None:
                executor.shutdown()
            if prev_threads is not None:
                torch.set_num_threads(prev_threads)
            
        return self.best_individual, self.best_fitness

    def extract_interpretability(self, trained_model, D_train):
        """
        7. Trích xuất khả năng diễn giải (Interpretability - Hậu xử lý)
        """
        # Need to run a forward pass so pykan can calculate node/edge scores
        print("Running forward pass to calculate activations...")
        _ = trained_model(D_train['train_input'])
        
        # Feature Selection & Importance
        print("Calculating feature scores...")
        feature_scores = trained_model.feature_score
        print(f"Feature scores: {feature_scores}")
        
        # Symbolic Formula Extraction
        print("Extracting symbolic formulas...")
        # auto_symbolic will try to fit basic math formulas to spline edges
        # and replace them with the one that has the highest R^2
        try:
            lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'cos', 'abs']
            trained_model.auto_symbolic(lib=lib)
            print("Symbolic formulas extracted successfully.")
        except Exception as e:
            print(f"Failed to run auto_symbolic: {e}")
            
        return trained_model, feature_scores
