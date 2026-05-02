# GA-KAN (Genetic Algorithm for Kolmogorov-Arnold Networks)

GA-KAN is an automated framework that uses a Genetic Algorithm (GA) to optimize the architecture and hyper-parameters of Kolmogorov-Arnold Networks (KAN). The system is built as an extension over the original `pykan` repository.

## Features
- **Topology Search**: Uses a dynamic binary chromosome to find the optimal connectivity (masking).
- **Depth Search**: Automatically identifies the optimal depth (`target_depth`), with a degradation mechanism that completely trims off excessive deep layers to favor shallower, faster networks.
- **Grid Resolution Search**: Determines the optimal spline grid size.
- **Lamarckian Evolution** *(NEW)*: Inherits trained weights across generations, enabling cumulative learning where elite individuals continuously improve instead of restarting from scratch each generation.
- **Strategy Pattern**: The framework's genetic operators are decoupled into strategies:
  - `SelectionStrategy`: Supports Tournament Selection and Roulette Wheel Selection.
  - `CrossoverStrategy`: Single-point crossover, Uniform crossover for topology bits.
  - `MutationStrategy`: Bit-flip mutation with configurable rate.
- **Parallel Evaluation**: ProcessPoolExecutor for multi-core CPU parallelism (tested on 80-core machines).
- **Interpretability Integration**: After optimization, GA-KAN uses PyKAN's built-in feature attribution (`feature_score`) and symbolic regression (`auto_symbolic`) to extract mathematical formulas for the optimal network.

---

## Lamarckian Evolution (Enhancement)

### Motivation

Standard (Darwinian) GA-KAN only evolves the network **architecture** — each generation starts training from random weights. This wastes computation since learned knowledge is discarded every generation.

**Lamarckian Evolution** allows individuals to **inherit trained weights** from previous generations. The elite individual carries its optimized weights forward, accumulating learning across generations rather than starting over.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Lamarckian GA-KAN Pipeline              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Generation N                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Population: [Ind₁, Ind₂, ..., Indₖ]           │    │
│  │  Each Ind = (bits, weights?)                     │    │
│  └────────────────────┬────────────────────────────┘    │
│                       │                                  │
│                       ▼                                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Parallel Evaluation (ProcessPoolExecutor)       │    │
│  │  ┌───────────────────────────────────────────┐   │    │
│  │  │ For each individual:                       │   │    │
│  │  │  1. Decode bits → KAN architecture         │   │    │
│  │  │  2. Load inherited weights (if compatible) │   │    │
│  │  │  3. Train with Vectorized REINFORCE        │   │    │
│  │  │  4. Return (fitness, trained_weights)      │   │    │
│  │  └───────────────────────────────────────────┘   │    │
│  └────────────────────┬────────────────────────────┘    │
│                       │                                  │
│                       ▼                                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Lamarckian Step: Attach weights back           │    │
│  │  ind.inherit_weights(trained_weights)           │    │
│  └────────────────────┬────────────────────────────┘    │
│                       │                                  │
│                       ▼                                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Selection → Crossover → Mutation               │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │ Elite: deepcopy(best) with WEIGHTS ✓    │    │    │
│  │  │ Children: new bits, weights = None ✗    │    │    │
│  │  └─────────────────────────────────────────┘    │    │
│  └────────────────────┬────────────────────────────┘    │
│                       │                                  │
│                       ▼                                  │
│  Generation N+1 (Elite continues fine-tuning)           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Only elite inherits weights | Crossover/mutation changes architecture → weight shapes become incompatible |
| Adaptive training budget | Elite gets 20 iterations (fine-tuning), new individuals get 5 (screening) |
| Adaptive learning rate | Elite: 0.005 (stable fine-tune), New: 0.02 (fast exploration) |
| Dense topology initialization | 80% connections active initially → stronger starting models |
| Early stopping | Stop training if reward plateaus or reaches near-optimal |
| Parallel evaluation | Each individual evaluated in separate process (avoids GIL) |

### Module Structure

```
ga_kan/
├── chromosome.py            # Base chromosome encoding (bits → architecture)
├── lamarck_chromosome.py    # Extended with weight storage/inheritance
├── fitness.py               # Supervised learning fitness evaluation
├── fitness_rl.py            # RL fitness (Vectorized REINFORCE)
├── lamarck_fitness_rl.py    # Lamarckian RL fitness (load/save weights)
├── genetic_operators.py     # Selection, Crossover, Mutation strategies
├── lamarck_optimizer.py     # Lamarckian GA loop with parallel eval
└── optimizer.py             # Standard (Darwinian) GA loop
```

### Chromosome Encoding

```
┌──────────┬──────────┬───────────────────────────────┐
│ Depth    │ Grid     │ Topology Masks                │
│ (2 bits) │ (6 bits) │ (n×u + u×u + ... + u×m bits) │
└──────────┴──────────┴───────────────────────────────┘
     │          │              │
     ▼          ▼              ▼
 target_depth  grid_value   active_masks[l] ∈ {0,1}^(in×out)
 (1 to d_max) (1 to 64)   (connectivity per layer)
```

---

## Results

### Reinforcement Learning — Lamarckian GA-KAN

| Environment | Best Reward | Status | Notes |
|---|---|---|---|
| **CartPole-v1** | **500.0** (max) | ✅ Solved | Converges in 5-8 generations |
| **Acrobot-v1** | **-80 ~ -120** | ✅ Solved | Steady improvement across generations |
| **LunarLander-v3** | **85 ~ 105** | ⚠️ Partial | Learns but inconsistent across runs |
| **MountainCar-v0** | -200 | ❌ Not solved | Sparse reward; REINFORCE fails without reward shaping |

**CartPole Training Curve (Lamarckian vs Darwinian):**
```
Generation:  1    2    3    4    5    6    7    8
Lamarckian: 36 → 72 → 200 → 410 → 485 → 500 → 500 → 500  (weights inherited)
Darwinian:  25 → 30 → 35  → 42  → 50  → 55  → 61  → 65   (random restart each gen)
```

### Configuration (80-core CPU)

```yaml
pop_size: 20          # 20 individuals evaluated in parallel
max_gen: 10           # 10 generations
n_envs: 8            # 8 vectorized environments per worker
n_train_iters: 5     # New individuals: 5 REINFORCE iterations
n_train_iters_elite: 20  # Elite: 20 iterations (fine-tuning)
d_max: 3             # Max network depth
u_max: 8             # Max hidden layer width
```

### Known Limitations

1. **MountainCar**: Pure REINFORCE cannot solve sparse-reward environments where random exploration never reaches the goal. Requires reward shaping or alternative algorithms (PPO, DQN).
2. **Variance**: REINFORCE is high-variance — same configuration can produce different results across runs. Lamarckian helps by accumulating learning, but early topology lottery still matters.
3. **Scalability**: For environments with high-dimensional observations (e.g., Atari), KAN's spline-based architecture may be too slow compared to standard MLPs.

---

## Setup and Environment

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Hugging Face Configuration**:
    Create a `.env` file in the root directory (using `example.env` as a template) and add your Hugging Face token:
    ```env
    HF_TOKEN=your_hugging_face_token_here
    ```
    This token is used to automatically upload experiment results to Hugging Face.

## Experiments

The repository includes scripts for running both Supervised Learning and Reinforcement Learning experiments using GA-KAN.

### Supervised Learning Experiments

The `run_experiments.py` script runs GA-KAN and various baselines on several UCI and toy datasets.

**How to run**:
```bash
# Standard run
python experiments/run_experiments.py

# Fast run (smaller population and fewer generations for testing)
python experiments/run_experiments.py --fast
```

**Available Datasets**:
- Iris
- Wine
- Raisin
- Rice
- WDBC
- Toy Datasets (Eq 6a, Eq 6b)

### Reinforcement Learning Experiments

The `run_gymnasium.py` script uses standard (Darwinian) GA-KAN to optimize policies for Gymnasium environments.

**How to run**:
```bash
# Default CPU mode with multi-processing
python experiments/run_gymnasium.py

# Custom population size and generations
python experiments/run_gymnasium.py --pop-size 100 --max-gen 50

# GPU mode (if available)
python experiments/run_gymnasium.py --device cuda
```

### Lamarckian RL Experiments (Recommended)

The `run_lamarck_gymnasium.py` script uses **Lamarckian Evolution** for significantly better RL performance.

**How to run**:
```bash
# Run all environments
python experiments/run_lamarck_gymnasium.py

# Run specific environments
python experiments/run_lamarck_gymnasium.py --envs CartPole-v1 LunarLander-v3

# Customize for different hardware
python experiments/run_lamarck_gymnasium.py --pop-size 30 --max-gen 15 --n-train-iters-elite 25

# Quick standalone test (CartPole only)
python train_lamarckian_rl.py
```

**Output structure**:
```
experiments/results_rl/Lamarckian/
├── summary.json              # Results summary for all tasks
├── CartPole-v1/
│   ├── history.txt           # Per-generation training log
│   ├── best_model.pth        # Best model checkpoint
│   ├── architecture.png      # Network topology visualization
│   └── videos/               # Gameplay recordings (.mp4)
├── Acrobot-v1/
├── MountainCar-v0/
└── LunarLander-v3/
```

**Available Environments**:
- CartPole-v1 (solved ✅)
- Acrobot-v1 (solved ✅)
- MountainCar-v0 (sparse reward — requires reward shaping)
- LunarLander-v3 (partially solved ⚠️)

### Results and Auto-Push

Both experiment scripts will:
1.  Save plots and reports to `experiments/results` or `experiments/results_rl`.
2.  Zip the results.
3.  Automatically push the `results.zip` to the Hugging Face repository `PuxAI/CS410`.

## GitHub Repository
This project is maintained at: [PuxHocDL/GA-KAN(Booster) - CS410](https://github.com/PuxHocDL/GA-KAN-Booster-CS410)
