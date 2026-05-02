import torch
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical
import numpy as np
import copy
import warnings

from .lamarck_chromosome import LamarckChromosome
from .fitness import build_optimal_model

# Import original training function to reuse logic
from .fitness_rl import train_rl_vectorized


class MountainCarRewardWrapper(gym.Wrapper):
    """Shaped reward for MountainCar: reward based on height (position) and velocity."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # position is obs[0], range [-1.2, 0.6]. Goal is 0.5+
        # velocity is obs[1], range [-0.07, 0.07]
        # Add shaped reward: encourage going right and high
        shaped = obs[0] + 0.5  # shift so that position 0.5 → reward 1.0
        shaped += abs(obs[1]) * 10  # reward velocity in any direction (need momentum)
        if terminated:  # reached goal
            shaped += 100.0
        reward = shaped
        return obs, reward, terminated, truncated, info

def evaluate_lamarckian_fitness_rl(individual: LamarckChromosome, env_name: str, N_steps=8, device='cpu',
                        envs=None, vectorization_mode='sync',
                        n_train_iterations=3, n_train_iterations_elite=10,
                        max_steps=500):
    """
    Evaluates fitness by running a Vectorized REINFORCE algorithm.
    
    Lamarckian Modification:
    - Loads weights into the model before training if individual has inherited weights.
    - Elite individuals (with weights) get MORE training iterations since they're fine-tuning.
    - New individuals get fewer iterations (just enough to assess architecture quality).
    - Returns the trained state_dict along with fitness to be inherited by the next generation.
    - Applies reward shaping for sparse-reward environments (MountainCar).
    """
    model = build_optimal_model(individual, device=device)
    
    # [Lamarckian Step] Load inherited weights if they exist and are compatible
    has_prior_weights = False
    if individual.has_weights():
        try:
            model.load_state_dict(individual.weights)
            has_prior_weights = True
        except Exception as e:
            warnings.warn(f"Failed to load weights for individual (likely due to shape mismatch from mutation/crossover): {e}")

    # Adaptive training: elite (with weights) gets more iterations
    actual_iterations = n_train_iterations_elite if has_prior_weights else n_train_iterations
    
    # Higher lr for new individuals (need to learn faster), lower for fine-tuning elite
    lr = 0.005 if has_prior_weights else 0.02
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    own_envs = envs is None
    if own_envs:
        # For MountainCar, use reward shaping wrapper
        if 'MountainCar' in env_name:
            envs = gym.make_vec(env_name, num_envs=N_steps, vectorization_mode=vectorization_mode,
                                wrappers=[MountainCarRewardWrapper])
        else:
            try:
                envs = gym.make_vec(env_name, num_envs=N_steps, vectorization_mode=vectorization_mode)
            except TypeError:
                envs = gym.make_vec(env_name, num_envs=N_steps)

    try:
        rewards = []
        for i in range(actual_iterations):
            r = train_rl_vectorized(model, envs, optimizer, device=device, max_steps=max_steps)
            rewards.append(r)
            # Early stop: if consistently near-optimal for bounded-reward envs,
            # or if improvement plateaus for negative-reward envs
            if len(rewards) >= 3:
                recent = rewards[-3:]
                # For positive-reward envs (CartPole): stop if near max
                if recent[-1] >= max_steps * 0.90 and recent[-2] >= max_steps * 0.90:
                    break
                # For any env: stop if no improvement over last 3 iters
                if max(recent) - min(recent) < 1.0 and i >= 5:
                    break
            
        avg_reward = max(rewards)
        
        # [Lamarckian Step] Extract the weights AFTER training
        # Move state dict to CPU to save GPU memory when storing in population
        trained_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
    except Exception:
        if own_envs:
            envs.close()
        return float('inf'), None

    if own_envs:
        envs.close()
        
    return -avg_reward, trained_weights
