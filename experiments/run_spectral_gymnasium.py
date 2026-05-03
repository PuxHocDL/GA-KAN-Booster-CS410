"""
Experiment runner for Spectral (Chebyshev) GA-KAN on Gymnasium RL tasks.

Evaluates the Spectral KAN approach against the B-spline baseline.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral_kan import (
    SpectralConfig, SpectralChromosome, SpectralGAOptimizer,
    SpectralCrossover, FrequencyMutation, TournamentSelection,
    ChebKAN
)
from spectral_kan.fitness_rl import build_spectral_model


# =====================================================================
# =====================================================================
# Per-task configuration (Baseline - No Novelty Search)
# =====================================================================
TASK_CONFIGS = {
    'CartPole-v1': {
        'obs_dim': 4, 'act_dim': 2,
        'max_steps': 500,
        'n_train_iterations': 5,
        'n_train_iterations_elite': 20,
        'pop_size': 20,
        'max_gen': 12,
        'N_steps': 8,
        'd_max': 4,
        'u_max': 16,
        'degree_max': 10,
    },
    'Acrobot-v1': {
        'obs_dim': 6, 'act_dim': 3,
        'max_steps': 500,
        'n_train_iterations': 8,
        'n_train_iterations_elite': 25,
        'pop_size': 24,
        'max_gen': 15,
        'N_steps': 8,
        'd_max': 4,
        'u_max': 20,
        'degree_max': 10,
    },
    'MountainCar-v0': {
        'obs_dim': 2, 'act_dim': 3,
        'max_steps': 200,
        'n_train_iterations': 15,
        'n_train_iterations_elite': 40,
        'pop_size': 30,
        'max_gen': 25,
        'N_steps': 16,
        'd_max': 4,
        'u_max': 16,
        'degree_max': 10,
    },
    'LunarLander-v3': {
        'obs_dim': 8, 'act_dim': 4,
        'max_steps': 500,
        'n_train_iterations': 8,
        'n_train_iterations_elite': 25,
        'pop_size': 24,
        'max_gen': 15,
        'N_steps': 8,
        'd_max': 4,
        'u_max': 24,
        'degree_max': 10,
    },
    'Pendulum-v1': {
        'obs_dim': 3, 'act_dim': 5,
        'max_steps': 200,
        'n_train_iterations': 15,
        'n_train_iterations_elite': 40,
        'pop_size': 30,
        'max_gen': 25,
        'N_steps': 16,
        'd_max': 4,
        'u_max': 12,
        'degree_max': 10,
    },
    'LunarLander-Wind': {
        'obs_dim': 8, 'act_dim': 4,
        'max_steps': 500,
        'n_train_iterations': 10,
        'n_train_iterations_elite': 30,
        'pop_size': 28,
        'max_gen': 20,
        'N_steps': 12,
        'd_max': 4,
        'u_max': 24,
        'degree_max': 10,
    },
}

# Novelty Search configs per environment (used with --novelty flag)
NOVELTY_CONFIGS = {
    'MountainCar-v0': {'novelty_weight': 0.3, 'novelty_k': 5},
    'Pendulum-v1': {'novelty_weight': 0.3, 'novelty_k': 5},
    'LunarLander-Wind': {'novelty_weight': 0.15, 'novelty_k': 5},
    # These don't really need novelty but can be tried:
    'CartPole-v1': {'novelty_weight': 0.1, 'novelty_k': 5},
    'Acrobot-v1': {'novelty_weight': 0.1, 'novelty_k': 5},
    'LunarLander-v3': {'novelty_weight': 0.1, 'novelty_k': 5},
}

# Experiment stages for incremental exploration
STAGES = {
    1: {
        'name': 'Baseline (4 original envs, no novelty)',
        'envs': ['CartPole-v1', 'Acrobot-v1', 'LunarLander-v3', 'MountainCar-v0'],
        'use_novelty': False,
    },
    2: {
        'name': 'Novelty Search on hard envs',
        'envs': ['MountainCar-v0'],
        'use_novelty': True,
    },
    3: {
        'name': 'New environments (Pendulum + LunarLander-Wind)',
        'envs': ['Pendulum-v1', 'LunarLander-Wind'],
        'use_novelty': False,
    },
    4: {
        'name': 'New environments + Novelty Search',
        'envs': ['Pendulum-v1', 'LunarLander-Wind'],
        'use_novelty': True,
    },
    5: {
        'name': 'All 6 environments + Novelty Search',
        'envs': ['CartPole-v1', 'Acrobot-v1', 'LunarLander-v3', 'MountainCar-v0', 'Pendulum-v1', 'LunarLander-Wind'],
        'use_novelty': True,
    },
}


def setup_logger(env_name, output_dir):
    """Set up logging for an experiment."""
    log_file = os.path.join(output_dir, f'{env_name}_spectral.log')
    logger = logging.getLogger(f'spectral_{env_name}')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    return logger


def evaluate_final_policy(individual: SpectralChromosome, env_name: str, 
                          n_episodes=100, device='cpu'):
    """Evaluate the final policy over many episodes to get reliable statistics."""
    from spectral_kan.fitness_rl import DiscretePendulumWrapper, PendulumTerminationWrapper
    
    model = build_spectral_model(individual, device=device)
    if model is None:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    
    if individual.has_weights():
        try:
            state_dict = {k: torch.from_numpy(v) for k, v in individual.weights.items()}
            model.load_state_dict(state_dict)
        except:
            pass
    
    model.eval()
    episode_rewards = []
    
    for _ in range(n_episodes):
        if 'Pendulum' in env_name:
            env = gym.make('Pendulum-v1')
            env = DiscretePendulumWrapper(env)
            env = PendulumTerminationWrapper(env)
        elif env_name == 'LunarLander-Wind':
            env = gym.make('LunarLander-v3', enable_wind=True, wind_power=15.0, turbulence_power=1.5)
        else:
            env = gym.make(env_name)
        
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                logits = model(state_t)
                action = logits.argmax(dim=-1).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        env.close()
    
    return {
        'mean': float(np.mean(episode_rewards)),
        'std': float(np.std(episode_rewards)),
        'min': float(np.min(episode_rewards)),
        'max': float(np.max(episode_rewards)),
        'episodes': n_episodes,
    }


def record_video(individual: SpectralChromosome, env_name: str, output_dir: str,
                 n_episodes=3, device='cpu'):
    """Record video of the best policy playing the environment."""
    from spectral_kan.fitness_rl import DiscretePendulumWrapper
    
    try:
        import moviepy  # noqa: F401
    except ImportError:
        print("  [skip] moviepy not installed, skipping video recording")
        return None
    
    model = build_spectral_model(individual, device=device)
    if model is None:
        return None
    
    if individual.has_weights():
        try:
            state_dict = {k: torch.from_numpy(v) for k, v in individual.weights.items()}
            model.load_state_dict(state_dict)
        except:
            return None
    
    model.eval()
    video_dir = os.path.join(output_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    
    # Create appropriate environment
    if 'Pendulum' in env_name:
        env = gym.make('Pendulum-v1', render_mode='rgb_array')
        env = DiscretePendulumWrapper(env)
    elif env_name == 'LunarLander-Wind':
        env = gym.make('LunarLander-v3', render_mode='rgb_array',
                       enable_wind=True, wind_power=15.0, turbulence_power=1.5)
    else:
        env = gym.make(env_name, render_mode='rgb_array')
    
    env = gym.wrappers.RecordVideo(
        env, video_dir,
        episode_trigger=lambda ep: True,
        name_prefix=f'spectral_{env_name}'
    )
    
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                logits = model(state_t)
                action = logits.argmax(dim=-1).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        print(f"  Video episode {ep+1}: reward = {total_reward:.1f}")
    
    env.close()
    return video_dir


def save_model(individual: SpectralChromosome, output_dir: str, env_name: str):
    """Save the full model (architecture + weights) for later loading."""
    depth, degree, width, masks = individual.decode()
    
    model_data = {
        'architecture': {
            'width': width,
            'degree': degree,
            'depth': depth,
        },
        'chromosome_bits': individual.bits.tolist(),
        'config': {
            'n': individual.config.n,
            'm': individual.config.m,
            'd_max': individual.config.d_max,
            'u_max': individual.config.u_max,
            'degree_max': individual.config.degree_max,
        },
    }
    
    # Save architecture info
    arch_file = os.path.join(output_dir, f'{env_name}_model_arch.json')
    with open(arch_file, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    # Save weights (as torch state_dict for easy loading)
    if individual.has_weights():
        weights_file = os.path.join(output_dir, f'{env_name}_spectral_weights.pt')
        torch_weights = {k: torch.from_numpy(v) for k, v in individual.weights.items()}
        torch.save(torch_weights, weights_file)
    
    return arch_file


def run_single_experiment(env_name: str, output_dir: str, num_workers=None, config_override=None):
    """Run spectral GA-KAN experiment on a single environment.
    
    Parameters
    ----------
    config_override : dict, optional
        If provided, use this config instead of TASK_CONFIGS[env_name].
        Useful for running with/without novelty on the same env.
    """
    cfg = config_override if config_override else TASK_CONFIGS[env_name]
    
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(env_name, output_dir)
    
    logger.info(f"{'='*60}")
    logger.info(f"Spectral GA-KAN Experiment: {env_name}")
    logger.info(f"{'='*60}")
    
    # Build config
    spectral_config = SpectralConfig(
        n=cfg['obs_dim'],
        m=cfg['act_dim'],
        d_max=cfg['d_max'],
        u_max=cfg['u_max'],
        degree_max=cfg['degree_max'],
    )
    
    logger.info(f"Config: obs={cfg['obs_dim']}, act={cfg['act_dim']}, "
                f"d_max={cfg['d_max']}, u_max={cfg['u_max']}, degree_max={cfg['degree_max']}")
    logger.info(f"Chromosome length: {spectral_config.b_total} bits")
    logger.info(f"Population: {cfg['pop_size']}, Generations: {cfg['max_gen']}")
    logger.info(f"Train iters: new={cfg['n_train_iterations']}, elite={cfg['n_train_iterations_elite']}")
    
    # Setup GA operators
    selection = TournamentSelection(tournament_size=3)
    crossover = SpectralCrossover(
        pc_topology=0.3,
        freq_cutoff_ratio=0.5,
        alpha=0.5
    )
    mutation = FrequencyMutation(
        pm_topology=0.02,
        pm_architecture=0.005,
        weight_mutation_rate=0.3,
        noise_scale=0.1,
        freq_cutoff_ratio=0.5
    )
    
    # Build optimizer
    novelty_weight = cfg.get('novelty_weight', 0.0)
    novelty_k = cfg.get('novelty_k', 5)
    
    optimizer = SpectralGAOptimizer(
        config=spectral_config,
        selection_strategy=selection,
        crossover_strategy=crossover,
        mutation_strategy=mutation,
        pop_size=cfg['pop_size'],
        max_gen=cfg['max_gen'],
        N_steps=cfg['N_steps'],
        device='cpu',
        n_train_iterations=cfg['n_train_iterations'],
        n_train_iterations_elite=cfg['n_train_iterations_elite'],
        max_steps_per_iter=cfg['max_steps'],
        dense_init=True,
        num_workers=num_workers,
        elitism_count=2,
        novelty_weight=novelty_weight,
        novelty_k=novelty_k,
    )
    
    variant_label = 'novelty' if novelty_weight > 0 else 'baseline'
    logger.info(f"Variant: {variant_label}")
    if novelty_weight > 0:
        logger.info(f"Novelty Search ENABLED: weight={novelty_weight}, k={novelty_k}")
    
    # Run GA
    start_time = time.time()
    best_individual, best_fitness = optimizer.run(env_name, logger=logger)
    elapsed = time.time() - start_time
    
    logger.info(f"\nOptimization complete in {elapsed:.1f}s")
    logger.info(f"Best GA fitness (neg reward): {best_fitness:.4f}")
    logger.info(f"Best GA reward: {-best_fitness:.4f}")
    
    # Architecture of best individual
    depth, degree, width, masks = best_individual.decode()
    logger.info(f"Best architecture: width={width}, degree={degree}")
    logger.info(f"Estimated params: {best_individual._estimate_params(width, degree)}")
    
    # Final evaluation
    logger.info(f"\nRunning final evaluation (100 episodes, greedy)...")
    eval_stats = evaluate_final_policy(best_individual, env_name, n_episodes=100)
    logger.info(f"Final Policy: mean={eval_stats['mean']:.2f} ± {eval_stats['std']:.2f}, "
                f"min={eval_stats['min']:.2f}, max={eval_stats['max']:.2f}")
    
    # Save full model (architecture + weights)
    logger.info(f"\nSaving model...")
    save_model(best_individual, output_dir, env_name)
    logger.info(f"Model saved to {output_dir}")
    
    # Record video
    logger.info(f"\nRecording video of best policy...")
    try:
        video_dir = record_video(best_individual, env_name, output_dir, n_episodes=3)
        if video_dir:
            logger.info(f"Videos saved to {video_dir}")
    except Exception as e:
        logger.info(f"Video recording failed (non-critical): {e}")
    
    # Save results
    results = {
        'env_name': env_name,
        'method': 'Spectral_GA_KAN',
        'best_ga_reward': float(-best_fitness),
        'final_eval': eval_stats,
        'architecture': {
            'width': width,
            'degree': degree,
            'depth': depth,
            'estimated_params': best_individual._estimate_params(width, degree),
        },
        'config': cfg,
        'time_seconds': elapsed,
        'history': optimizer.history,
    }
    
    results_file = os.path.join(output_dir, f'{env_name}_spectral_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Spectral analysis
    model = build_spectral_model(best_individual)
    if model is not None and best_individual.has_weights():
        try:
            state_dict = {k: torch.from_numpy(v) for k, v in best_individual.weights.items()}
            model.load_state_dict(state_dict)
            summary = model.spectral_summary()
            logger.info(f"\nSpectral Energy Distribution:")
            for layer_info in summary:
                energy_str = ', '.join([f'T{i}={e:.4f}' for i, e in enumerate(layer_info['energy_per_degree'])])
                logger.info(f"  Layer {layer_info['layer']} ({layer_info['shape']}): [{energy_str}]")
            
            # Save architecture visualization
            from spectral_kan.visualization import plot_cheb_kan, plot_spectral_energy
            arch_plot_path = os.path.join(output_dir, f'{env_name}_architecture.png')
            plot_cheb_kan(model, title=f"Spectral GA-KAN Policy ({env_name})", save_path=arch_plot_path)
            
            energy_plot_path = os.path.join(output_dir, f'{env_name}_spectral_energy.png')
            plot_spectral_energy(model, title=f"Spectral Energy ({env_name})", save_path=energy_plot_path)
        except Exception as e:
            logger.info(f"Visualization failed (non-critical): {e}")
    
    return results


def run_stage(stage_num, num_workers=None, output_base=None):
    """Run a specific experiment stage."""
    if stage_num not in STAGES:
        print(f"Invalid stage {stage_num}. Available: {list(STAGES.keys())}")
        return {}
    
    stage = STAGES[stage_num]
    use_novelty = stage['use_novelty']
    
    if output_base is None:
        suffix = '_novelty' if use_novelty else ''
        output_base = os.path.join(os.path.dirname(__file__), f'results_spectral{suffix}')
    
    print(f"\n{'='*60}")
    print(f"STAGE {stage_num}: {stage['name']}")
    print(f"Environments: {stage['envs']}")
    print(f"Novelty Search: {'ON' if use_novelty else 'OFF'}")
    print(f"Output: {output_base}")
    print(f"{'='*60}")
    
    all_results = {}
    for env_name in stage['envs']:
        cfg = TASK_CONFIGS[env_name].copy()
        
        # Apply novelty config if enabled
        if use_novelty and env_name in NOVELTY_CONFIGS:
            cfg.update(NOVELTY_CONFIGS[env_name])
        
        output_dir = os.path.join(output_base, env_name)
        print(f"\nStarting: {env_name} ({'+ novelty' if use_novelty else 'baseline'})")
        
        try:
            results = run_single_experiment(env_name, output_dir, num_workers=num_workers, config_override=cfg)
            all_results[env_name] = results
        except Exception as e:
            print(f"FAILED: {env_name} - {e}")
            import traceback
            traceback.print_exc()
            all_results[env_name] = {'error': str(e)}
    
    # Save combined results
    combined_file = os.path.join(output_base, f'stage{stage_num}_results.json')
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    _print_summary(all_results, stage['name'])
    return all_results


def run_all_experiments(num_workers=None, output_base=None):
    """Run spectral GA-KAN on all configured environments (baseline, no novelty)."""
    if output_base is None:
        output_base = os.path.join(os.path.dirname(__file__), 'results_spectral')
    
    all_results = {}
    envs_to_run = list(TASK_CONFIGS.keys())
    
    for env_name in envs_to_run:
        output_dir = os.path.join(output_base, env_name)
        print(f"\n{'='*60}")
        print(f"Starting: {env_name}")
        print(f"{'='*60}")
        
        try:
            results = run_single_experiment(env_name, output_dir, num_workers=num_workers)
            all_results[env_name] = results
        except Exception as e:
            print(f"FAILED: {env_name} - {e}")
            import traceback
            traceback.print_exc()
            all_results[env_name] = {'error': str(e)}
    
    # Save combined results
    combined_file = os.path.join(output_base, 'all_results.json')
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    _print_summary(all_results, "Spectral GA-KAN (Baseline)")
    return all_results


def _print_summary(all_results, title):
    """Print a formatted summary table."""
    print(f"\n{'='*60}")
    print(f"SUMMARY - {title}")
    print(f"{'='*60}")
    for env_name, res in all_results.items():
        if 'error' in res:
            print(f"  {env_name}: FAILED - {res['error']}")
        else:
            eval_stats = res['final_eval']
            arch = res['architecture']
            print(f"  {env_name}: {eval_stats['mean']:.1f} ± {eval_stats['std']:.1f} "
                  f"(arch: {arch['width']}, deg={arch['degree']}, "
                  f"params={arch['estimated_params']}, time={res['time_seconds']:.0f}s)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Spectral GA-KAN RL Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages (incremental exploration):
  1 - Baseline: CartPole, Acrobot, LunarLander, MountainCar (no novelty)
  2 - Novelty Search: MountainCar only (compare with stage 1)
  3 - New envs baseline: Pendulum, LunarLander-Wind (no novelty)
  4 - New envs + novelty: Pendulum, LunarLander-Wind (with novelty)
  5 - Full run: all 6 environments with novelty search

Examples:
  python run_spectral_gymnasium.py --stage 1            # baseline 4 envs
  python run_spectral_gymnasium.py --stage 2            # novelty on MountainCar
  python run_spectral_gymnasium.py --env MountainCar-v0 --novelty  # single env + novelty
  python run_spectral_gymnasium.py --env Pendulum-v1    # single env baseline
"""
    )
    parser.add_argument('--env', type=str, default=None,
                        help='Specific environment to run')
    parser.add_argument('--stage', type=int, default=None,
                        help='Run a predefined stage (1-5)')
    parser.add_argument('--novelty', action='store_true',
                        help='Enable Novelty Search for the selected env(s)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    args = parser.parse_args()
    
    if args.stage:
        # Run a specific stage
        run_stage(args.stage, num_workers=args.workers, output_base=args.output)
    elif args.env:
        # Run a single environment
        cfg = TASK_CONFIGS[args.env].copy()
        if args.novelty and args.env in NOVELTY_CONFIGS:
            cfg.update(NOVELTY_CONFIGS[args.env])
        
        suffix = '_novelty' if args.novelty else ''
        output_dir = args.output or os.path.join(
            os.path.dirname(__file__), f'results_spectral{suffix}', args.env
        )
        run_single_experiment(args.env, output_dir, num_workers=args.workers, config_override=cfg)
    else:
        # Default: run stage 1 (baseline)
        run_stage(1, num_workers=args.workers, output_base=args.output)
