import numpy as np
import copy
from .chromosome import Chromosome, ChromosomeConfig

class LamarckChromosome(Chromosome):
    def __init__(self, config: ChromosomeConfig, bits=None, weights=None):
        super().__init__(config, bits)
        # For Lamarckian evolution, we store the state_dict (weights) of the trained model
        self.weights = copy.deepcopy(weights) if weights is not None else None

    def inherit_weights(self, weights):
        """
        Save the weights after training so they can be passed to the next generation.
        """
        self.weights = copy.deepcopy(weights)

    def has_weights(self):
        return self.weights is not None
