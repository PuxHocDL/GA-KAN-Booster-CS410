"""
Runner for Spectral GA-KAN experiments.

Usage:
    python modal_runner_spectral.py                    # Run all environments
    python modal_runner_spectral.py --env CartPole-v1  # Run single env
    python modal_runner_spectral.py --workers 16       # Specify worker count
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run_spectral_gymnasium import run_all_experiments, run_single_experiment

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Spectral GA-KAN RL Experiments')
    parser.add_argument('--env', type=str, default=None,
                        help='Specific environment (default: all). Options: CartPole-v1, Acrobot-v1, LunarLander-v3, MountainCar-v0')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto based on CPU count)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: experiments/results_spectral/)')
    args = parser.parse_args()

    if args.env:
        output_dir = args.output or os.path.join('experiments', 'results_spectral', args.env)
        run_single_experiment(args.env, output_dir, num_workers=args.workers)
    else:
        run_all_experiments(num_workers=args.workers, output_base=args.output)
