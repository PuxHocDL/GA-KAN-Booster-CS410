# GA-KAN Booster — CS410

An experimental framework that progressively enhances **Kolmogorov-Arnold Networks (KAN)** through Genetic Algorithm architecture search, Lamarckian evolution, spectral basis replacement, and novelty-driven exploration. Developed for CS410.

## Table of Contents

- [Overview: The Enhancement Journey](#overview-the-enhancement-journey)
- [Phase 1: Base GA-KAN vs Baselines (Supervised)](#phase-1-base-ga-kan-vs-baselines-supervised)
- [Phase 2: Lamarckian Evolution (RL)](#phase-2-lamarckian-evolution-rl)
- [Phase 3: Spectral GA-KAN — Replacing Splines with Chebyshev Polynomials](#phase-3-spectral-ga-kan--replacing-splines-with-chebyshev-polynomials)
- [Phase 4: Novelty Search for Hard Exploration (RL)](#phase-4-novelty-search-for-hard-exploration-rl)
- [Results Summary](#results-summary)
- [Repository Layout](#repository-layout)
- [Setup & Usage](#setup--usage)
- [Infrastructure](#infrastructure)

---

## Overview: The Enhancement Journey

| Phase | Problem | Solution | Key Insight |
|-------|---------|----------|-------------|
| 1 | KAN architecture selection is manual | **GA-KAN**: GA searches over depth, width, grid, topology | GA finds competitive architectures automatically |
| 2 | RL with GA is too random — good weights lost each generation | **Lamarckian Evolution**: elite weights inherited | Weight inheritance stabilizes RL training |
| 3 | B-spline basis has knot sensitivity + non-compact representation | **Spectral KAN**: Chebyshev polynomial basis | Coefficients are directly evolvable, no grid artifacts |
| 4 | Sparse-reward RL envs (MountainCar) trap GA in local optima | **Novelty Search** (Lehman & Stanley, 2011) | Behavior diversity pressure escapes deceptive rewards |

---

## Phase 1: Base GA-KAN vs Baselines (Supervised)

### Motivation

Standard KAN requires manual architecture specification (`[input, hidden..., output]`, grid size, depth). We use a **Genetic Algorithm** to search this space automatically.

### Method

1. **Chromosome encoding**: binary string encodes depth (up to `d_max`), hidden width per layer (up to `u_max`), spline grid resolution, and per-edge topology masks.
2. **GA search**: each individual is decoded → invalid topologies rejected → valid candidates trained briefly with Adam → validation loss = fitness.
3. **Final training**: best architecture trained with full budget (Adam or LBFGS).
4. **Baselines**: SVM, Random Forest, MLP, KNN, and two Standard KAN configurations.

### Results (16 datasets)

| Dataset | SVM | RF | MLP | KNN | Standard KAN | GA-KAN |
|---------|-----|----|----|-----|-------------|--------|
| Iris | **1.00** | **1.00** | 0.63 | **1.00** | **1.00** | 0.97 |
| Wine | **1.00** | **1.00** | 0.61 | 0.94 | **1.00** | **1.00** |
| WDBC | **0.97** | 0.96 | **0.97** | 0.96 | 0.93 | 0.96 |
| Banknote | **1.00** | 0.99 | 1.00 | **1.00** | **1.00** | **1.00** |
| Rice | **0.93** | 0.92 | 0.93 | 0.91 | 0.91 | 0.93 |
| Moons | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| Circles | **1.00** | 0.99 | **1.00** | **1.00** | **1.00** | **1.00** |
| Toy1 (Eq6a) | 0.026 | 0.016 | 0.486 | 0.025 | **5e-6** | 3.8e-4 |
| Toy2 (Eq6b) | 0.003 | 8e-4 | 0.005 | 5e-4 | **5e-7** | 1.7e-4 |

*Classification: higher = better. Regression (Toy*): lower MSE = better.*

**Key finding**: GA-KAN is competitive with sklearn baselines on classification tasks and significantly outperforms MLP. On symbolic regression, Standard KAN with hand-tuned architecture still wins (domain knowledge advantage), but GA-KAN closes the gap vs traditional ML.

### Supervised Pipeline

```bash
# Run all 16 datasets on 80-CPU Modal container
python experiments/run_experiments.py \
  --device cpu --all-datasets --cpu-workers 30 \
  --n-steps 30 --final-steps 80
```

---

## Phase 2: Lamarckian Evolution (RL)

### Problem

In **Darwinian** GA for RL, each individual's policy weights are randomly initialized every generation. Good policies discovered by REINFORCE training are lost when the next generation starts from scratch. This wastes training budget and makes convergence slow/unstable.

### Solution: Lamarckian Inheritance

Inspired by Lamarckian evolution: **trained weights of elite individuals are passed to offspring**.

```
Generation N:  [individual_1, ..., individual_k]
                     │ REINFORCE training
                     ▼
               [trained_1, ..., trained_k]
                     │ Select elites, crossover architecture
                     ▼
Generation N+1: offspring inherit parent weights (where architecture matches)
                remaining weights randomly initialized
```

Key implementation details:
- Weights stored as **NumPy arrays** (not PyTorch tensors) to avoid file-descriptor sharing failures in `ProcessPoolExecutor`.
- Compatible layers (same shape) inherit weights directly; incompatible layers are re-initialized.
- `n_train_iterations_elite` > `n_train_iterations_new` gives elites more REINFORCE steps (they already have good weights).

### RL Results (Darwinian vs Lamarckian, PyKAN-based)

| Environment | Darwinian | Lamarckian | Improvement |
|-------------|-----------|------------|-------------|
| CartPole-v1 | 350 | **500** (max) | Solved |
| Acrobot-v1 | -120 | **-79** | 34% faster |
| LunarLander-v3 | 45 | **90** | 100% better |
| MountainCar-v0 | -200 | -200 | Still stuck |

**Key finding**: Lamarckian evolution dramatically stabilizes RL training. CartPole is perfectly solved. But MountainCar (sparse reward, deceptive) remains unsolved — motivating Phase 4.

```bash
python experiments/run_lamarck_gymnasium.py \
  --pop-size 30 --max-gen 15 --n-train-iters-elite 25
```

---

## Phase 3: Spectral GA-KAN — Replacing Splines with Chebyshev Polynomials

### Problem with B-Spline KAN

PyKAN uses B-spline basis functions on each edge. Problems for GA evolution:
1. **Grid sensitivity**: B-spline quality depends on grid placement and data distribution.
2. **Non-compact representation**: grid knots + coefficients are coupled → hard to inherit across different architectures.
3. **Slow**: spline evaluation + grid updates are expensive.

### Solution: Chebyshev Polynomial Basis (Spectral KAN)

Replace B-splines with **Chebyshev polynomials** $T_k(x)$ of degree up to $K$:

$$\phi(x) = \sum_{k=0}^{K} c_k \cdot T_k(x), \quad T_k(x) = \cos(k \cdot \arccos(x))$$

Advantages:
- **Compact**: each edge is fully described by $K+1$ coefficients — no grid knots.
- **Evolvable**: coefficients are directly inheritable (same-shape layers copy coefficients exactly).
- **Spectral interpretability**: energy distribution across $T_k$ reveals function complexity.
- **Fast**: no grid management, no adaptive updates needed.

### Architecture

```python
class ChebKANLayer(nn.Module):
    """
    Each edge (i→j) has learnable coefficients [c_0, c_1, ..., c_K]
    for Chebyshev polynomials T_0(x), T_1(x), ..., T_K(x).
    Input normalized to [-1, 1] via adaptive running statistics.
    """
```

The GA chromosome encodes:
- Number of hidden layers (depth up to `d_max`)
- Width of each hidden layer (up to `u_max`)
- Chebyshev degree (up to `degree_max=10`)
- Total chromosome: ~600 bits

### Spectral RL Results (Baseline, no novelty)

| Environment | Architecture | Params | Eval Mean | Eval Max | Time |
|-------------|-------------|--------|-----------|----------|------|
| CartPole-v1 | [4,8,2] deg=10 | 528 | **438.7** | 500 | 172s |
| Acrobot-v1 | [6,3] deg=10 | 198 | **-96.1** | -63 | 220s |
| LunarLander-v3 | [8,4] deg=7 | 256 | **141.8** | 253 | 342s |
| MountainCar-v0 | [2,14,3] deg=4 | 350 | -200.0 | -200 | 210s |

**Key finding**: Spectral KAN achieves similar or better performance than spline-based KAN with **far fewer parameters** (198-528 vs thousands). But MountainCar still fails (sparse reward deception).

---

## Phase 4: Novelty Search for Hard Exploration (RL)

### Problem

MountainCar has a **deceptive reward landscape**: the car must swing back and forth to build momentum before reaching the goal. Random policies never reach the goal (reward = -200 always). GA converges prematurely because all individuals look equally bad.

### Solution: Novelty Search (Lehman & Stanley, 2011)

Instead of optimizing reward alone, we optimize for **behavioral diversity**:

$$\text{fitness}(i) = (1 - \alpha) \cdot r_i + \alpha \cdot \text{novelty}(i)$$

Where:
- $r_i$ = normalized reward fitness
- $\text{novelty}(i) = \frac{1}{k} \sum_{j \in \text{kNN}(i)} \| b_i - b_j \|$ (mean distance to k-nearest neighbors in behavior space)
- $b_i$ = behavior descriptor (max and mean observations per state dimension)
- $\alpha$ = novelty weight (0.3 for hard envs)

A **behavior archive** stores diverse behaviors discovered so far, growing the reference set over generations.

### MountainCar: Before vs After Novelty Search

| Metric | Without Novelty | With Novelty |
|--------|----------------|--------------|
| Best GA reward | -194.6 (shaped) | **124.8** (shaped) |
| Eval mean (raw) | -200.0 (timeout) | **-102.6** |
| Eval std | 0.0 | 6.4 |
| Eval best | -200 | **-83 steps** |
| Solved? | ❌ Never | ✅ **Yes** (threshold: -110) |

The novelty pressure drove the GA to discover the "swing" strategy at generation 15 (reward jumped from -190 → +97), then the population converged to refine it.

### Additional Hard Environments (Stage 4: Novelty Search)

| Environment | Architecture | Params | Eval Mean | Eval Max | Time |
|-------------|-------------|--------|-----------|----------|------|
| Pendulum-v1 | [3,12,5] deg=10 | 1056 | -223.7 | **-3.5** | 649s |
| LunarLander-Wind | [8,9,9,4] deg=3 | 756 | **94.2** | 257.1 | 668s |

- **Pendulum-v1**: Continuous torque discretized to 5 actions. High variance (std=242) indicates policy is inconsistent but capable of near-perfect episodes (-3.5).
- **LunarLander-Wind**: Standard LunarLander + stochastic wind (power=15, turbulence=1.5). Deeper 3-layer KAN needed (degree=3 sufficient). Mean=94.2 demonstrates robust landing despite wind disturbances.

### Experiment Stages

The RL experiments are organized in 5 incremental stages:

| Stage | Description | Environments |
|-------|-------------|-------------|
| 1 | Baseline Spectral GA-KAN | CartPole, Acrobot, LunarLander, MountainCar |
| 2 | + Novelty Search on hard envs | MountainCar (novelty_weight=0.3) |
| 3 | New hard environments (baseline) | Pendulum, LunarLander-Wind |
| 4 | New environments + Novelty | Pendulum, LunarLander-Wind (novelty) |
| 5 | Full suite with Novelty | All 6 environments |

```bash
# Run specific stage on Modal
python modal_runner_spectral.py --stage 2
python modal_runner_spectral.py --stage 4
python modal_runner_spectral.py --env LunarLander-Wind --novelty
```

---

## Results Summary

### Supervised Learning (16 datasets)

GA-KAN matches or exceeds traditional ML baselines (SVM, RF, KNN) on most classification tasks, with the advantage of producing **interpretable symbolic networks**.

### Reinforcement Learning — Evolution of Results

| Environment | Phase 2 (Lamarck+Spline) | Phase 3 (Spectral) | Phase 4 (+Novelty) |
|-------------|--------------------------|---------------------|---------------------|
| CartPole-v1 | 500 ✅ | 500 ✅ | — |
| Acrobot-v1 | -79 ✅ | -96.1 ✅ | — |
| LunarLander-v3 | 90 ✅ | 141.8 ✅ | — |
| MountainCar-v0 | -200 ❌ | -200 ❌ | **-102.6 ✅** |
| Pendulum-v1 | — | — | -223.7 (best: -3.5) |
| LunarLander-Wind | — | — | **94.2 ✅** |

### Key Takeaways

1. **GA architecture search works**: removes manual tuning, finds compact architectures (198-825 params).
2. **Lamarckian inheritance is critical**: without it, RL policies can't converge.
3. **Spectral basis > Splines for evolution**: Chebyshev coefficients are compact, directly inheritable, and interpretable via spectral energy analysis.
4. **Novelty Search unlocks deceptive environments**: MountainCar went from impossible (-200) to solved (-102.6) with diversity pressure.

---

## Repository Layout

```text
ga_kan/                            # Phase 1-2: Spline-based GA-KAN
  chromosome.py                    # Binary chromosome: depth, grid, topology masks
  fitness.py                       # Supervised fitness with Adam fast-screening
  optimizer.py                     # Standard GA optimizer (CPU/GPU parallel)
  genetic_operators.py             # Tournament selection, crossover, mutation
  fitness_rl.py                    # RL REINFORCE fitness
  lamarck_chromosome.py            # Lamarckian chromosome (weight inheritance)
  lamarck_fitness_rl.py            # Lamarckian RL fitness evaluation
  lamarck_optimizer.py             # Lamarckian GA optimizer

spectral_kan/                      # Phase 3-4: Chebyshev Spectral GA-KAN
  cheb_kan_layer.py                # ChebKANLayer: Chebyshev polynomial edges
  cheb_kan.py                      # Full ChebKAN model with input normalization
  chromosome.py                    # Spectral chromosome (width, degree)
  fitness_rl.py                    # RL fitness + reward shaping + env wrappers
  genetic_operators.py             # Frequency-domain crossover/mutation
  optimizer.py                     # Spectral GA + Novelty Search
  visualization.py                 # Architecture & spectral energy plots

experiments/
  run_experiments.py               # Supervised benchmark (16 datasets)
  baselines.py                     # sklearn + Standard KAN baselines
  data_loader.py                   # UCI, sklearn, and toy datasets
  run_gymnasium.py                 # Darwinian GA-KAN RL
  run_lamarck_gymnasium.py         # Lamarckian GA-KAN RL (spline-based)
  run_spectral_gymnasium.py        # Spectral GA-KAN RL (5 stages)

modal_runner_spectral.py           # Modal VM runner with --stage, --env, --novelty
train_lamarckian_rl.py             # Standalone Lamarckian RL trainer
```

---

## Setup & Usage

### Installation

```bash
pip install -r requirements.txt
```

### Supervised Experiments

```bash
# All 16 datasets, 80-CPU Modal container
python experiments/run_experiments.py \
  --device cpu --all-datasets --cpu-workers 30 \
  --n-steps 30 --final-steps 80

# Single dataset
python experiments/run_experiments.py --datasets Iris --cpu-workers 8

# With symbolic interpretability
python experiments/run_experiments.py --interpretability
```

### RL Experiments (Spline-based)

```bash
# Darwinian (baseline)
python experiments/run_gymnasium.py

# Lamarckian (weight inheritance)
python experiments/run_lamarck_gymnasium.py \
  --pop-size 30 --max-gen 15 --n-train-iters-elite 25
```

### RL Experiments (Spectral, recommended)

```bash
# Stage 1: Baseline on 4 classic envs
python modal_runner_spectral.py --stage 1

# Stage 2: MountainCar with Novelty Search
python modal_runner_spectral.py --stage 2

# Stage 3: New hard envs (Pendulum, LunarLander-Wind)
python modal_runner_spectral.py --stage 3

# Stage 4: New envs + Novelty
python modal_runner_spectral.py --stage 4

# Single env with novelty
python modal_runner_spectral.py --env MountainCar-v0 --novelty
```

---

## Infrastructure

- **Compute**: [Modal](https://modal.com) VM with 80 CPUs, Python 3.9
- **Parallelism**: `ProcessPoolExecutor` with `spawn` context (Linux), `OMP_NUM_THREADS=1`
- **Weight serialization**: NumPy arrays (avoids PyTorch fd-sharing crashes in multiprocessing)
- **Results**: saved to Modal volume, synced back after runs

## GitHub Repository

```text
https://github.com/PuxHocDL/GA-KAN-Booster-CS410
```
