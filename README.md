# GA-KAN Booster for CS410

GA-KAN Booster is an experimental framework for searching Kolmogorov-Arnold
Network (KAN) architectures with a Genetic Algorithm (GA). The project extends
PyKAN with supervised-learning experiments, CPU-parallel GA evaluation,
standard ML baselines, standard KAN baselines, and reinforcement-learning
experiments.

The current supervised runner is optimized for CPU-only Modal containers. It
uses multiple CPU workers for GA fitness evaluation, trains candidate networks
briefly during search, then trains the best discovered GA-KAN architecture in a
separate final phase.

## Highlights

- GA-KAN architecture search over depth, spline grid size, and topology masks.
- CPU parallel fitness evaluation with `ProcessPoolExecutor` or thread fallback.
- Multiprocessing-safe dataset handling: process workers receive NumPy-backed
  data at startup, avoiding PyTorch tensor file-descriptor sharing failures.
- Adam-based GA search fitness for faster screening.
- Adam final training for GA-KAN by default, with optional LBFGS.
- Configurable final training budget, baseline KAN budget, depth, width, worker
  count, and dataset selection.
- Interpretability and symbolic extraction are available, but skipped by default
  because `auto_symbolic` can be very slow on wide or deep KANs.
- Comparison table printed at the end in `Dataset | Model | Score` format.
- Optional run subdirectories to avoid overwriting results when multiple
  containers write to the same Modal volume.

## Repository Layout

```text
ga_kan/
  chromosome.py            # Binary chromosome: depth, grid, topology masks
  fitness.py               # Supervised GA-KAN fitness evaluation
  optimizer.py             # Standard GA optimizer with CPU/GPU evaluation modes
  genetic_operators.py     # Tournament selection, crossover, mutation
  fitness_rl.py            # RL fitness helpers
  lamarck_*.py             # Lamarckian RL extensions

experiments/
  run_experiments.py       # Supervised benchmark runner
  baselines.py             # sklearn and Standard KAN baselines
  data_loader.py           # UCI, sklearn, and toy datasets
  run_gymnasium.py         # Darwinian GA-KAN RL experiments
  run_lamarck_gymnasium.py # Lamarckian GA-KAN RL experiments
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional Hugging Face upload support uses `HF_TOKEN` from `.env`:

```env
HF_TOKEN=your_hugging_face_token_here
```

If `HF_TOKEN` is missing, the experiment still runs; the final upload step is
skipped or reports the missing token.

## Supervised Method

`experiments/run_experiments.py` evaluates GA-KAN against sklearn baselines and
two Standard KAN baselines.

The supervised pipeline is:

1. Load and normalize a dataset.
2. Build a chromosome search space:
   - `d_max`: maximum KAN depth to search.
   - `u_max`: maximum hidden width per hidden layer.
   - `grid`: spline grid resolution encoded by chromosome bits.
   - topology masks: binary edge masks per layer.
3. Run GA search:
   - each individual is decoded into a KAN architecture.
   - invalid disconnected topologies are rejected.
   - each valid candidate is trained briefly with Adam for `--n-steps`.
   - validation loss is used as fitness.
4. Train the best GA-KAN architecture again in a final phase:
   - Adam by default.
   - LBFGS available with `--final-optimizer lbfgs`.
5. Run sklearn baselines:
   - SVM
   - Random Forest
   - MLP, with sklearn `max_iter=500`
   - KNN
6. Run two Standard KAN baselines:
   - `[d, 2d+1, C]`
   - `[d, 5, 5, 5, C]`
7. Save results and print a comparison table.

### Training Budgets

The main knobs are:

```text
--n-steps              Adam steps per candidate during GA search
--final-steps          final training steps for the selected GA-KAN model
--kan-baseline-steps   training steps for Standard KAN baselines
```

By default:

```text
normal mode: n_steps=12, final_steps=50, kan_baseline_steps=final_steps
fast mode:   n_steps=5,  final_steps=12, kan_baseline_steps=final_steps
```

This distinction matters. GA search evaluates many models, so `--n-steps` is a
screening budget. `--final-steps` is used only after the best architecture has
been selected. Standard KAN baselines default to the same step count as
GA-KAN final training, so KAN-vs-KAN comparisons are not accidentally biased by
different final training budgets.

The rough GA search cost per dataset is:

```text
pop_size * max_gen * n_steps
```

For example, `--pop-size 30 --max-gen 30 --n-steps 30` means 27,000 short
candidate-training steps before final training and baselines.

## Recommended CPU Commands

The Modal web terminal used for this project is CPU-only. If `nvidia-smi` is not
available, use `--device cpu`.

Balanced CPU run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/run_experiments.py \
  --device cpu \
  --cpu-workers 30 \
  --n-steps 30 \
  --final-steps 80
```

Fast smoke test:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/run_experiments.py \
  --device cpu \
  --fast \
  --datasets Iris \
  --cpu-workers 8 \
  --no-upload
```

Higher-quality GA-KAN final training:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/run_experiments.py \
  --device cpu \
  --cpu-workers 30 \
  --n-steps 50 \
  --final-steps 120 \
  --final-lr 0.01
```

Use LBFGS for final GA-KAN training instead of Adam:

```bash
python experiments/run_experiments.py \
  --device cpu \
  --cpu-workers 30 \
  --n-steps 30 \
  --final-steps 80 \
  --final-optimizer lbfgs
```

Run a single dataset:

```bash
python experiments/run_experiments.py \
  --device cpu \
  --datasets Iris \
  --cpu-workers 30
```

Run the extended dataset suite:

```bash
python experiments/run_experiments.py \
  --device cpu \
  --all-datasets \
  --cpu-workers 30
```

Run `Digits` with a lighter architecture search:

```bash
python experiments/run_experiments.py \
  --device cpu \
  --all-datasets \
  --datasets Digits \
  --cpu-workers 30 \
  --n-steps 12 \
  --final-steps 50 \
  --d-max 4 \
  --u-max 6
```

Enable symbolic interpretability only when needed:

```bash
python experiments/run_experiments.py \
  --device cpu \
  --cpu-workers 30 \
  --n-steps 30 \
  --final-steps 80 \
  --interpretability
```

## Dataset Suites

Default supervised suite:

```text
Iris
Wine
Raisin
Rice
WDBC
Toy1_Eq_6a
Toy2_Eq_6b
```

Extended supervised suite, enabled with `--all-datasets`:

```text
Iris
Wine
WDBC
Digits
Raisin
Rice
Banknote
Seeds
Glass
Moons
Circles
Toy1_Eq_6a
Toy2_Eq_6b
Toy3_sincos
Toy4_radial
Diabetes
```

`Digits` is much slower than small tabular datasets because it has 64 input
features and 10 classes. That increases the number of possible topology edges
and makes each KAN forward/backward pass heavier.

## CLI Reference for `run_experiments.py`

| Argument | Default | Meaning |
|---|---:|---|
| `--device` | `cpu` | Device used for GA-KAN and Standard KAN training. |
| `--fast` | off | Smaller quick-test configuration. |
| `--all-datasets` | off | Use the extended dataset suite. |
| `--datasets` | none | Comma-separated dataset filter, for example `Iris,Wine`. |
| `--pop-size` | `30` | GA population size in normal mode. |
| `--max-gen` | `30` | Number of GA generations in normal mode. |
| `--n-steps` | `12` | Adam steps used to evaluate each GA candidate. |
| `--final-steps` | `50` | Final training steps for the selected GA-KAN. |
| `--final-optimizer` | `adam` | `adam` or `lbfgs` for final GA-KAN training. |
| `--final-lr` | optimizer default | Learning rate for final GA-KAN training. |
| `--kan-baseline-steps` | `final_steps` | Steps for Standard KAN baselines. |
| `--d-max` | `5` | Maximum depth searched by GA-KAN. |
| `--u-max` | `10` | Maximum hidden width per hidden layer. |
| `--cpu-workers` | `min(pop_size, cpu_count)` | Parallel CPU workers for GA fitness. |
| `--cpu-torch-threads` | `1` | PyTorch intra-op threads per worker. |
| `--parallel-backend` | `auto` | `process`, `thread`, or platform-aware `auto`. |
| `--run-name` | none | Writes to `experiments/results/<run-name>/`. |
| `--interpretability` | off | Runs feature scores, symbolic extraction, and plotting. |
| `--skip-interpretability` | on | Explicitly skip interpretability. |
| `--no-upload` | off | Skip zipping and Hugging Face upload. |

## Output and Volume Safety

By default, supervised results are written to:

```text
experiments/results/results.csv
experiments/results/<Dataset>/interpretability_report.txt
experiments/results/<Dataset>/ga_kan_architecture.png
```

When multiple Modal containers share the same sync volume and write to the same
path, later runs can overwrite earlier files. Use `--run-name` for separate
outputs:

```bash
python experiments/run_experiments.py \
  --device cpu \
  --cpu-workers 30 \
  --run-name run_cpu_30x30_adam
```

This writes to:

```text
experiments/results/run_cpu_30x30_adam/results.csv
experiments/results/run_cpu_30x30_adam/<Dataset>/
```

The runner prints a final table:

```text
Comparison Results:
Dataset                        Model    Score
   Iris                          SVM        1
   Iris                           RF        1
   Iris                          MLP 0.633333
   Iris                          KNN        1
   Iris    Standard KAN [d, 2d+1, C] 0.766667
   Iris Standard KAN [d, 5, 5, 5, C]      0.3
   Iris                       GA-KAN        1
```

## CPU Parallel Evaluation Details

The optimizer uses CPU parallelism for CPU runs. On Linux, `--parallel-backend
auto` resolves to `process`. On Windows, it resolves to `thread` to avoid slow
process startup.

For process mode, worker processes initialize their own tensor copies from
NumPy data once at startup. This avoids the common PyTorch multiprocessing
failure where tensors are sent through queues using shared file descriptors:

```text
BlockingIOError: [Errno 11] Resource temporarily unavailable
concurrent.futures.process.BrokenProcessPool
```

If a container still has strict process/socket limits, switch to threads:

```bash
python experiments/run_experiments.py \
  --device cpu \
  --parallel-backend thread \
  --cpu-workers 30
```

## Interpretability

Interpretability is disabled by default because PyKAN `auto_symbolic` can be
very expensive. For deep or wide models it may print many lines like:

```text
fixing (l,i,j) with sin, r2=...
```

and can dominate runtime. Enable it only for final analysis:

```bash
python experiments/run_experiments.py --interpretability
```

When enabled, the runner saves feature scores, symbolic formulas, and plots for
the selected GA-KAN model.

## Reinforcement Learning Experiments

The repository also contains RL experiments:

```bash
python experiments/run_gymnasium.py
python experiments/run_lamarck_gymnasium.py
```

`run_lamarck_gymnasium.py` implements Lamarckian evolution for RL policies,
where compatible elite weights can be carried forward across generations. This
is separate from the supervised CPU runner described above.

Common RL options include:

```bash
python experiments/run_lamarck_gymnasium.py \
  --pop-size 30 \
  --max-gen 15 \
  --n-train-iters-elite 25
```

## Troubleshooting

No GPU available:

```text
[GPU] error: No such file or directory: 'nvidia-smi'
```

Use:

```bash
python experiments/run_experiments.py --device cpu
```

`Digits` is slow:

Use fewer steps or smaller KAN search space:

```bash
python experiments/run_experiments.py \
  --all-datasets \
  --datasets Digits \
  --n-steps 12 \
  --final-steps 50 \
  --d-max 4 \
  --u-max 6
```

Symbolic extraction is too slow:

Do not pass `--interpretability`. It is skipped by default.

Multiple containers overwrite results:

Use a different `--run-name` per container.

## GitHub Repository

This project is maintained at:

```text
https://github.com/PuxHocDL/GA-KAN-Booster---CS410
```
