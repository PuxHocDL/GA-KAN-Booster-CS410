"""
Fitness evaluation for Spectral KAN in RL tasks.

Uses REINFORCE with vectorized environments.
Builds ChebKAN models from SpectralChromosome and evaluates policy quality.
"""

import torch
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical
import numpy as np
import copy
import warnings

from .chromosome import SpectralChromosome, is_valid_spectral_topology
from .cheb_kan import ChebKAN


class MountainCarRewardWrapper(gym.Wrapper):
    """Potential-based reward shaping for MountainCar.
    
    Uses energy-based potential: potential = height + 0.5 * velocity^2
    Reward shaping: F = gamma * phi(s') - phi(s) + bonus for reaching goal.
    This is provably policy-invariant (Ng et al., 1999).
    """
    def __init__(self, env):
        super().__init__(env)
        self._prev_potential = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_potential = self._potential(obs)
        return obs, info
    
    def _potential(self, obs):
        # Position: obs[0] in [-1.2, 0.6], Velocity: obs[1] in [-0.07, 0.07]
        height = np.sin(3 * obs[0])  # Actual terrain height
        kinetic = 50.0 * obs[1] ** 2  # Kinetic energy (scaled)
        return height + kinetic
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_potential = self._potential(obs)
        # Potential-based shaping (gamma ≈ 1.0)
        shaping = current_potential - self._prev_potential
        self._prev_potential = current_potential
        
        # Base reward: -1 per step + shaping + big bonus for goal
        shaped_reward = reward + 10.0 * shaping
        if terminated:  # Reached goal (position >= 0.5)
            shaped_reward += 200.0
        
        return obs, shaped_reward, terminated, truncated, info


def build_spectral_model(individual: SpectralChromosome, device='cpu'):
    """
    Build a ChebKAN model from a SpectralChromosome.
    
    Returns
    -------
    ChebKAN model or None if topology is invalid.
    """
    depth, degree, width, active_masks = individual.decode()
    
    # Validate topology
    if not is_valid_spectral_topology(active_masks):
        return None
    
    # Convert masks to tensors
    mask_tensors = [
        torch.tensor(mask, dtype=torch.float32, device=device)
        for mask in active_masks
    ]
    
    try:
        model = ChebKAN(
            width=width,
            degree=degree,
            masks=mask_tensors,
            input_normalize='tanh'
        )
        model = model.to(device)
    except Exception as e:
        warnings.warn(f"Failed to build ChebKAN with width={width}, degree={degree}: {e}")
        return None
    
    return model


def train_rl_vectorized(model, envs, optimizer, gamma=0.99, device='cpu', max_steps=500):
    """
    REINFORCE training on vectorized environment.
    Identical logic to ga_kan version but works with any model.
    """
    num_envs = envs.num_envs
    states, _ = envs.reset()
    
    log_prob_buf = []
    reward_buf = []
    done_buf = []
    finished = np.zeros(num_envs, dtype=bool)
    
    for _ in range(max_steps):
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
        logits = model(state_tensor)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        next_states, rews, terminated, truncated, _ = envs.step(actions.detach().cpu().numpy())
        dones = terminated | truncated
        
        log_prob_buf.append(log_probs)
        reward_buf.append(rews.astype(np.float32))
        done_buf.append(dones.astype(np.float32))
        
        finished |= dones
        states = next_states
        if finished.all():
            break
    
    if not log_prob_buf:
        return 0.0
    
    # Stack into (T, num_envs)
    log_probs_t = torch.stack(log_prob_buf, dim=0)
    rewards_t = torch.as_tensor(np.stack(reward_buf, axis=0), dtype=torch.float32, device=device)
    dones_t = torch.as_tensor(np.stack(done_buf, axis=0), dtype=torch.float32, device=device)
    
    # Active mask: transitions before first episode end
    prev_done_cum = torch.cumsum(dones_t, dim=0) - dones_t
    active = (prev_done_cum < 1.0).float()
    
    total_rewards = (rewards_t * active).sum(dim=0).detach().cpu().numpy()
    
    # Vectorised discounted returns
    masked_rewards = rewards_t * active
    T = masked_rewards.shape[0]
    returns_t = torch.zeros_like(masked_rewards)
    R = torch.zeros(num_envs, device=device)
    
    for t in range(T - 1, -1, -1):
        R = masked_rewards[t] + gamma * R * (1.0 - dones_t[t])
        returns_t[t] = R
    
    # Normalize returns
    flat_returns = returns_t[active.bool()]
    if flat_returns.numel() > 1:
        returns_t = (returns_t - flat_returns.mean()) / (flat_returns.std() + 1e-8)
    
    # Policy gradient loss
    loss = -(log_probs_t * returns_t * active).sum() / active.sum().clamp(min=1)
    
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return float(total_rewards.mean())


def evaluate_spectral_fitness_rl(individual: SpectralChromosome, env_name: str,
                                  N_steps=8, device='cpu',
                                  envs=None, vectorization_mode='sync',
                                  n_train_iterations=5, n_train_iterations_elite=15,
                                  max_steps=500):
    """
    Evaluate fitness of a SpectralChromosome via REINFORCE.
    
    Lamarckian:
    - Loads inherited Chebyshev coefficients if available
    - Elite (with weights) gets more training iterations
    - Returns trained weights for next generation
    
    Returns
    -------
    tuple: (fitness, trained_weights)
        fitness is negative reward (minimization), weights is state_dict
    """
    model = build_spectral_model(individual, device=device)
    if model is None:
        return float('inf'), None
    
    # [Lamarckian] Load inherited weights (stored as numpy, convert to torch)
    has_prior_weights = False
    if individual.has_weights():
        try:
            state_dict = {k: torch.from_numpy(v) for k, v in individual.weights.items()}
            model.load_state_dict(state_dict)
            has_prior_weights = True
        except Exception as e:
            pass  # Architecture changed, can't load weights
    
    # Adaptive training iterations
    actual_iterations = n_train_iterations_elite if has_prior_weights else n_train_iterations
    lr = 0.005 if has_prior_weights else 0.02
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    own_envs = envs is None
    if own_envs:
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
            
            # Early stopping
            if len(rewards) >= 3:
                recent = rewards[-3:]
                if recent[-1] >= max_steps * 0.90 and recent[-2] >= max_steps * 0.90:
                    break
                if max(recent) - min(recent) < 1.0 and i >= 5:
                    break
        
        avg_reward = max(rewards) if rewards else 0.0
        
        # [Lamarckian] Extract trained weights as numpy (avoids torch fd-sharing in multiprocessing)
        trained_weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        
    except Exception as e:
        if own_envs:
            envs.close()
        return float('inf'), None
    
    if own_envs:
        envs.close()
    
    # Fitness = negative reward (we minimize)
    fitness = -avg_reward
    return fitness, trained_weights
