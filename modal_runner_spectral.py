"""
Runner for Spectral GA-KAN experiments.

Usage:
    python modal_runner_spectral.py                        # Stage 1: baseline (4 original envs)
    python modal_runner_spectral.py --stage 2              # Stage 2: MountainCar + Novelty Search
    python modal_runner_spectral.py --stage 3              # Stage 3: new envs (Pendulum, LunarLander-Wind)
    python modal_runner_spectral.py --stage 4              # Stage 4: new envs + Novelty
    python modal_runner_spectral.py --env CartPole-v1      # Single env baseline
    python modal_runner_spectral.py --env MountainCar-v0 --novelty  # Single env + novelty
    python modal_runner_spectral.py --workers 16           # Specify worker count
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run_spectral_gymnasium import (
    run_all_experiments, run_single_experiment, run_stage,
    TASK_CONFIGS, NOVELTY_CONFIGS
)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Spectral GA-KAN RL Experiments')
    parser.add_argument('--env', type=str, default=None,
                        help='Specific environment. Options: CartPole-v1, Acrobot-v1, LunarLander-v3, MountainCar-v0, Pendulum-v1, LunarLander-Wind')
    parser.add_argument('--stage', type=int, default=None,
                        help='Run predefined stage (1=baseline, 2=novelty MountainCar, 3=new envs, 4=new+novelty, 5=all+novelty)')
    parser.add_argument('--novelty', action='store_true',
                        help='Enable Novelty Search')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto based on CPU count)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: experiments/results_spectral/)')
    args = parser.parse_args()

    if args.stage:
        run_stage(args.stage, num_workers=args.workers, output_base=args.output)
    elif args.env:
        cfg = TASK_CONFIGS[args.env].copy()
        if args.novelty and args.env in NOVELTY_CONFIGS:
            cfg.update(NOVELTY_CONFIGS[args.env])
        
        suffix = '_novelty' if args.novelty else ''
        output_dir = args.output or os.path.join('experiments', f'results_spectral{suffix}', args.env)
        run_single_experiment(args.env, output_dir, num_workers=args.workers, config_override=cfg)
    else:
        # Default: stage 1 (baseline)
        run_stage(1, num_workers=args.workers, output_base=args.output)
