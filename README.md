# GA-KAN (Genetic Algorithm for Kolmogorov-Arnold Networks)

GA-KAN is an automated framework that uses a Genetic Algorithm (GA) to optimize the architecture and hyper-parameters of Kolmogorov-Arnold Networks (KAN). The system is built as an extension over the original `pykan` repository.

## Features
- **Topology Search**: Uses a dynamic binary chromosome to find the optimal connectivity (masking).
- **Depth Search**: Automatically identifies the optimal depth (`target_depth`), with a degradation mechanism that completely trims off excessive deep layers to favor shallower, faster networks.
- **Grid Resolution Search**: Determines the optimal spline grid size.
- **Strategy Pattern**: The framework's genetic operators are decoupled into strategies:
  - `SelectionStrategy`: Supports Tournament Selection and Roulette Wheel Selection.
  - `CrossoverStrategy`: Single-point crossover for hyperparameters, pointwise for topology bits.
  - `MutationStrategy`: Bit-flip mutation.
- **Interpretability Integration**: After optimization, GA-KAN uses PyKAN's built-in feature attribution (`feature_score`) and symbolic regression (`auto_symbolic`) to extract mathematical formulas for the optimal network.

## Architecture & Code Structure

- `ga_kan/chromosome.py`: Contains the `Chromosome` and `ChromosomeConfig` for encoding/decoding lengths, grid values, target depth, and graph topologies. Also implements `is_valid_topology(masks)` to filter out disconnected subnetworks.
- `ga_kan/genetic_operators.py`: Implements GA evolutionary strategies.
- `ga_kan/fitness.py`: Evaluates an individual by instantiating the sub-network dynamically through pykan's `MultKAN(width=[...])` and applying zero-masks via `model.act_fun[l].mask`. Trains using LBFGS.
- `ga_kan/optimizer.py`: Contains `GAKANOptimizer` which loops through generations, keeping track of elites and executing the GA steps. Also includes `extract_interpretability()`.
- `test_ga_kan.py`: A demonstration on a synthetic `sklearn` dataset.

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

The `run_gymnasium.py` script uses GA-KAN to optimize policies for various Gymnasium environments.

**How to run**:
```bash
# Default CPU mode with multi-processing
python experiments/run_gymnasium.py

# Custom population size and generations
python experiments/run_gymnasium.py --pop-size 100 --max-gen 50

# GPU mode (if available)
python experiments/run_gymnasium.py --device cuda
```

**Available Environments**:
- CartPole-v1
- Acrobot-v1
- MountainCar-v0
- LunarLander-v3

### Results and Auto-Push

Both experiment scripts will:
1.  Save plots and reports to `experiments/results` or `experiments/results_rl`.
2.  Zip the results.
3.  Automatically push the `results.zip` to the Hugging Face repository `PuxAI/CS410`.

## GitHub Repository
This project is maintained at: [PuxHocDL/GA-KAN(Booster) - CS410](https://github.com/PuxHocDL/GA-KAN-Booster-CS410)
