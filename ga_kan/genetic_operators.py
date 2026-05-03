import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List, Tuple
from .chromosome import Chromosome, ChromosomeConfig

# ===============================
# Selection Strategy
# ===============================
class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, population: List[Chromosome], fitnesses: List[float], num_parents: int) -> List[Chromosome]:
        pass

class TournamentSelection(SelectionStrategy):
    def __init__(self, tournament_size=3):
        self.tournament_size = tournament_size
        
    def select(self, population: List[Chromosome], fitnesses: List[float], num_parents: int) -> List[Chromosome]:
        parents = []
        for _ in range(num_parents):
            tournament_size = min(self.tournament_size, len(population))
            indices = np.random.choice(len(population), size=tournament_size, replace=False)
            best_idx = min(indices, key=lambda idx: fitnesses[idx])
            parents.append(population[best_idx])
        return parents

class RouletteWheelSelection(SelectionStrategy):
    def select(self, population: List[Chromosome], fitnesses: List[float], num_parents: int) -> List[Chromosome]:
        # Since we are minimizing loss, we invert the fitness
        max_fit = max(fitnesses)
        inverted_fitnesses = [max_fit - f + 1e-6 for f in fitnesses] # avoid zero
        total_fit = sum(inverted_fitnesses)
        probs = [f / total_fit for f in inverted_fitnesses]
        
        indices = np.random.choice(len(population), size=num_parents, p=probs, replace=True)
        return [population[i] for i in indices]

# ===============================
# Crossover Strategy
# ===============================
class CrossoverStrategy(ABC):
    @abstractmethod
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        pass

class GAKANCrossover(CrossoverStrategy):
    def __init__(self, pc=0.5):
        """
        pc: probability of crossover for topology bits.
        """
        self.pc = pc
        
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        config = parent1.config
        
        c1_bits = np.copy(parent1.bits)
        c2_bits = np.copy(parent2.bits)
        
        # 1. Single-point crossover for Depth + Grid
        # Length of depth+grid prefix
        prefix_len = config.b_depth_len + config.b_grid_len
        crossover_point = random.randint(1, prefix_len - 1)
        
        c1_bits[crossover_point:prefix_len] = parent2.bits[crossover_point:prefix_len]
        c2_bits[crossover_point:prefix_len] = parent1.bits[crossover_point:prefix_len]
        
        # 2. Pointwise crossover for Topology
        for i in range(prefix_len, len(c1_bits)):
            if random.random() < self.pc:
                # Swap bits
                temp = c1_bits[i]
                c1_bits[i] = c2_bits[i]
                c2_bits[i] = temp
                
        return Chromosome(config, bits=c1_bits), Chromosome(config, bits=c2_bits)

class UniformCrossover(CrossoverStrategy):
    def __init__(self, crossover_rate=0.5):
        """
        crossover_rate: probability of swapping each bit position.
        """
        self.crossover_rate = crossover_rate

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        config = parent1.config

        c1_bits = np.copy(parent1.bits)
        c2_bits = np.copy(parent2.bits)

        for i in range(len(c1_bits)):
            if random.random() < self.crossover_rate:
                c1_bits[i], c2_bits[i] = c2_bits[i], c1_bits[i]

        return Chromosome(config, bits=c1_bits), Chromosome(config, bits=c2_bits)

# ===============================
# Mutation Strategy
# ===============================
class MutationStrategy(ABC):
    @abstractmethod
    def mutate(self, individual: Chromosome):
        pass

class BitFlipMutation(MutationStrategy):
    def __init__(self, pm=0.01):
        self.pm = pm
        
    def mutate(self, individual: Chromosome):
        for i in range(len(individual.bits)):
            if random.random() < self.pm:
                individual.bits[i] = 1 - individual.bits[i]
