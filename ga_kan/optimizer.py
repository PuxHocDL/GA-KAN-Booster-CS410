import copy
from typing import Dict
from tqdm import tqdm
import concurrent.futures
from .chromosome import Chromosome, ChromosomeConfig
from .genetic_operators import SelectionStrategy, CrossoverStrategy, MutationStrategy
from .fitness import evaluate_fitness, build_optimal_model

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
        device: str = 'cpu'
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
        
        self.population = []
        self.best_individual = None
        self.best_fitness = float('inf')
        
    def initialize_population(self):
        self.population = [Chromosome(self.config) for _ in range(self.pop_size)]
        
    def run(self, D_train: Dict, D_val: Dict):
        """
        Main Pipeline (Vòng lặp GA)
        """
        self.initialize_population()
        
        pbar = tqdm(range(self.max_gen), desc="GA Optimization", unit="gen")
        for gen in pbar:
            # Evaluate fitness in parallel to maximize GPU utilization safely
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 16)) as executor:
                futures = [executor.submit(evaluate_fitness, ind, D_train, D_val, self.task_type, self.N_steps, self.device) for ind in self.population]
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
