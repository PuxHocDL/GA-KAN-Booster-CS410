import torch
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical
import numpy as np

from .chromosome import Chromosome
from .fitness import build_optimal_model

# Speed knobs for tiny-model RL on big GPUs (reduce launch overhead, allow TF32).
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


def train_rl_vectorized(model, envs, optimizer, gamma=0.99, device='cpu', max_steps=500):
    """
    Trains the policy on a vectorized environment using a fully tensorised
    REINFORCE update. No per-env Python loops in the hot path.

    - Collects up to `max_steps` transitions across all envs simultaneously.
    - Stops early when every env has finished at least one episode.
    - Returns are computed with a vectorised reverse cumulative discount.
    """
    num_envs = envs.num_envs
    states, _ = envs.reset()

    log_prob_buf = []          # list[Tensor(num_envs,)]
    reward_buf = []            # list[np.ndarray(num_envs,)]
    done_buf = []              # list[np.ndarray(num_envs,)]  -- step-level done (for masking)
    finished = np.zeros(num_envs, dtype=bool)  # has env finished its FIRST episode?

    for _ in range(max_steps):
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
        logits = model(state_tensor)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)  # (num_envs,)

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

    # Mask transitions that occurred AFTER each env's first episode ended.
    # active[t, i] = 1 while env i has not yet been done up to step t (inclusive of the done step).
    # cumulative dones BEFORE step t:
    prev_done_cum = torch.cumsum(dones_t, dim=0) - dones_t
    active = (prev_done_cum < 1.0).float()  # 1 until (and including) first done step

    # Per-env first-episode return (sum of rewards while active) -- for logging
    total_rewards = (rewards_t * active).sum(dim=0).detach().cpu().numpy()

    # Vectorised discounted returns within first episode for each env (reverse cumsum).
    masked_rewards = rewards_t * active
    T = masked_rewards.shape[0]
    returns_t = torch.zeros_like(masked_rewards)
    R = torch.zeros(num_envs, device=device)
    # Build returns in reverse; this Python loop is over T (~500), not over envs.
    for t in range(T - 1, -1, -1):
        R = masked_rewards[t] + gamma * R * (1.0 - dones_t[t])
        returns_t[t] = R

    # Per-env baseline (mean over its active steps) for variance reduction.
    counts = active.sum(dim=0).clamp_min(1.0)
    mean_per_env = (returns_t * active).sum(dim=0) / counts
    var_per_env = ((returns_t - mean_per_env) ** 2 * active).sum(dim=0) / counts
    std_per_env = var_per_env.clamp_min(1e-8).sqrt()
    advantages = (returns_t - mean_per_env) / std_per_env

    policy_loss = -(log_probs_t * advantages * active).sum() / counts.sum()

    optimizer.zero_grad(set_to_none=True)
    policy_loss.backward()
    optimizer.step()

    return float(np.mean(total_rewards))


def evaluate_fitness_rl(individual: Chromosome, env_name: str, N_steps=10, device='cpu',
                        envs=None, vectorization_mode='async'):
    """
    Evaluates fitness by running a Vectorized REINFORCE algorithm for N_steps environments simultaneously.
    Fitness is the negative moving average reward.

    If `envs` is provided, the caller is responsible for its lifecycle (recommended:
    create once per env_name and reuse across the population to avoid repeated
    process-spawn cost when using AsyncVectorEnv).
    """
    model = build_optimal_model(individual, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    own_envs = envs is None
    if own_envs:
        try:
            envs = gym.make_vec(env_name, num_envs=N_steps, vectorization_mode=vectorization_mode)
        except TypeError:
            # Older gymnasium API
            envs = gym.make_vec(env_name, num_envs=N_steps)

    try:
        avg_reward = train_rl_vectorized(model, envs, optimizer, device=device)
    except Exception:
        if own_envs:
            envs.close()
        return float('inf')

    if own_envs:
        envs.close()
    return -avg_reward
