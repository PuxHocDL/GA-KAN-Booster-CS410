# GA-KAN framework
from .chromosome import Chromosome, ChromosomeConfig
from .genetic_operators import (
    SelectionStrategy, TournamentSelection, RouletteWheelSelection,
    CrossoverStrategy, GAKANCrossover, UniformCrossover,
    MutationStrategy, BitFlipMutation
)
from .fitness import evaluate_fitness, build_optimal_model
from .fitness_rl import evaluate_fitness_rl, train_rl_vectorized
from .lamarck_chromosome import LamarckChromosome
from .lamarck_fitness_rl import evaluate_lamarckian_fitness_rl
from .lamarck_optimizer import LamarckGAKANOptimizer
from .optimizer import GAKANOptimizer
