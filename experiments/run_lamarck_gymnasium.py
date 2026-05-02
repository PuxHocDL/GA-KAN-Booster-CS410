"""
Run Lamarckian GA-KAN Evolution across all Gymnasium RL tasks.
Saves training history, best model checkpoints, and gameplay videos.
"""
import argparse
import sys
import os
import time
import json
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ga_kan.chromosome import ChromosomeConfig
from ga_kan.genetic_operators import TournamentSelection, UniformCrossover, BitFlipMutation
from ga_kan.lamarck_optimizer import LamarckGAKANOptimizer
from ga_kan.fitness import build_optimal_model
from ga_kan.fitness_rl import train_rl_vectorized


def plot_architecture(model, save_dir, env_name):
    """Plot KAN architecture as a node-edge diagram showing layer widths and connections."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract architecture info
    width = model.width  # list of layer widths e.g. [4, 8, 2]
    n_layers = len(width)
    
    fig, ax = plt.subplots(1, 1, figsize=(max(6, n_layers * 2.5), max(4, max(width) * 0.6)))
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-0.5, max(width) - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Lamarck GA-KAN Architecture: {env_name}\nWidth: {width}", fontsize=12, fontweight='bold')
    
    # Node positions
    positions = {}  # (layer, node_idx) -> (x, y)
    for l in range(n_layers):
        n_nodes = width[l]
        y_offset = (max(width) - n_nodes) / 2
        for i in range(n_nodes):
            positions[(l, i)] = (l, y_offset + i)
    
    # Draw edges (connections based on masks)
    for l in range(n_layers - 1):
        if l < len(model.act_fun):
            mask = model.act_fun[l].mask.data.cpu().numpy()
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i, j] > 0.5:
                        x0, y0 = positions[(l, i)]
                        x1, y1 = positions[(l + 1, j)]
                        ax.plot([x0, x1], [y0, y1], 'b-', alpha=0.3, linewidth=0.5)
    
    # Draw nodes
    for l in range(n_layers):
        n_nodes = width[l]
        for i in range(n_nodes):
            x, y = positions[(l, i)]
            color = '#4CAF50' if l == 0 else ('#FF5722' if l == n_layers - 1 else '#2196F3')
            circle = plt.Circle((x, y), 0.15, color=color, zorder=5)
            ax.add_patch(circle)
    
    # Layer labels
    labels = ['Input'] + [f'Hidden {i}' for i in range(1, n_layers - 1)] + ['Output']
    for l in range(n_layers):
        ax.text(l, -1, f"{labels[l]}\n({width[l]})", ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'architecture.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Architecture plot saved: {save_dir}/architecture.png")


def setup_logger(name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(log_dir, 'history.txt'))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def record_video(model, env_name, video_dir, device='cpu', num_episodes=3):
    """Record gameplay video of the trained policy."""
    os.makedirs(video_dir, exist_ok=True)
    try:
        record_env = gym.make(env_name, render_mode="rgb_array")
        record_env = gym.wrappers.RecordVideo(
            record_env, video_folder=video_dir,
            name_prefix=f"{env_name}_lamarck",
            episode_trigger=lambda x: x < num_episodes
        )

        for ep in range(num_episodes):
            state, _ = record_env.reset()
            done = False
            truncated = False
            total_reward = 0
            while not (done or truncated):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits = model(state_tensor)
                    action = torch.argmax(logits, dim=1).item()
                state, reward, done, truncated, _ = record_env.step(action)
                total_reward += reward
            print(f"  Video episode {ep+1}: reward = {total_reward:.1f}")

        record_env.close()
        print(f"  Videos saved in {video_dir}")
    except Exception as e:
        print(f"  Warning: Failed to record video: {e}")


def run_single_task(env_name, n_in, m_out, args, base_output_dir):
    """Run Lamarckian evolution on a single RL task."""
    print(f"\n{'='*60}")
    print(f"  Lamarckian GA-KAN: {env_name} (obs={n_in}, act={m_out})")
    print(f"{'='*60}")

    task_dir = os.path.join(base_output_dir, env_name)
    logger = setup_logger(f"Lamarck_{env_name}", task_dir)

    config = ChromosomeConfig(n=n_in, m=m_out, d_max=args.d_max, u_max=args.u_max)

    # Per-task exploration settings
    # Harder tasks need higher mutation (more topology diversity)
    # and larger tournament size creates more selection pressure
    task_difficulty = getattr(args, 'task_difficulty', 'normal')
    if task_difficulty == 'hard':
        mutation_rate = 0.08  # Higher mutation for more exploration
        tournament_size = 2   # Lower pressure = more diversity preserved
        crossover_rate = 0.9  # More recombination
    else:
        mutation_rate = 0.03
        tournament_size = 3
        crossover_rate = 0.7

    selection = TournamentSelection(tournament_size=tournament_size)
    crossover = UniformCrossover(crossover_rate=crossover_rate)
    mutation = BitFlipMutation(pm=mutation_rate)

    optimizer = LamarckGAKANOptimizer(
        config=config,
        selection_strategy=selection,
        crossover_strategy=crossover,
        mutation_strategy=mutation,
        pop_size=args.pop_size,
        max_gen=args.max_gen,
        N_steps=args.n_envs,
        device=args.device,
        n_train_iterations=args.n_train_iters,
        n_train_iterations_elite=args.n_train_iters_elite,
        max_steps_per_iter=args.max_steps,
        vectorization_mode='sync',
        dense_init=True,
        num_workers=args.num_workers
    )

    start_time = time.time()
    best_ind, best_fitness = optimizer.run(env_name=env_name, logger=logger)
    elapsed = time.time() - start_time

    best_reward = -best_fitness
    logger.info(f"=== {env_name} Finished in {elapsed:.1f}s ===")
    logger.info(f"Best Reward: {best_reward:.4f}")
    logger.info(f"Best Architecture Depth: {best_ind.decode()[0]}")

    # Build final model with best weights
    best_model = build_optimal_model(best_ind, device=args.device)
    if best_ind.has_weights():
        best_model.load_state_dict(best_ind.weights)

    # Final evaluation (10 extra training iters on best architecture)
    print(f"\n  Final fine-tuning on best architecture...")
    opt = torch.optim.Adam(best_model.parameters(), lr=0.005)
    try:
        eval_envs = gym.make_vec(env_name, num_envs=args.n_envs, vectorization_mode='sync')
        final_rewards = []
        for _ in range(10):
            r = train_rl_vectorized(best_model, eval_envs, opt, device=args.device, max_steps=args.max_steps)
            final_rewards.append(r)
            if r >= args.max_steps * 0.95:
                break
        eval_envs.close()
        final_avg = sum(final_rewards) / len(final_rewards)
        final_max = max(final_rewards)
        print(f"  Final: avg={final_avg:.2f}, max={final_max:.2f}")
        logger.info(f"Final fine-tuning: avg={final_avg:.2f}, max={final_max:.2f}")
    except Exception as e:
        print(f"  Warning: Final eval failed: {e}")
        final_avg = best_reward
        final_max = best_reward

    # Save checkpoint
    ckpt_path = os.path.join(task_dir, 'best_model.pth')
    torch.save({
        'bits': best_ind.bits.tolist(),
        'state_dict': best_model.state_dict(),
        'fitness': best_fitness,
        'best_reward': best_reward,
        'final_avg_reward': final_avg,
        'final_max_reward': final_max,
        'elapsed_seconds': elapsed,
        'env_name': env_name,
        'architecture_depth': int(best_ind.decode()[0]),
    }, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    # Record video
    print(f"  Recording gameplay video...")
    video_dir = os.path.join(task_dir, 'videos')
    record_video(best_model, env_name, video_dir, device=args.device, num_episodes=3)

    # Save training curve plot
    try:
        plot_architecture(best_model, task_dir, env_name)
    except Exception as e:
        print(f"  Warning: Failed to plot architecture: {e}")

    return {
        'env_name': env_name,
        'best_reward': best_reward,
        'final_avg_reward': final_avg,
        'final_max_reward': final_max,
        'elapsed_seconds': elapsed,
        'architecture_depth': int(best_ind.decode()[0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Lamarckian GA-KAN for Gymnasium RL tasks")
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--pop-size', type=int, default=20)
    parser.add_argument('--max-gen', type=int, default=10)
    parser.add_argument('--n-envs', type=int, default=8,
                        help='Envs per worker (more = lower variance per iteration)')
    parser.add_argument('--n-train-iters', type=int, default=5,
                        help='Training iterations for new individuals')
    parser.add_argument('--n-train-iters-elite', type=int, default=20,
                        help='Training iterations for elite (with inherited weights)')
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--d-max', type=int, default=3)
    parser.add_argument('--u-max', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Worker processes (default=pop_size, max 60)')
    parser.add_argument('--envs', type=str, nargs='+', default=None,
                        help='Specific envs to run (e.g. CartPole-v1 LunarLander-v3)')
    args = parser.parse_args()

    # All RL tasks: (env_name, obs_dim, action_dim, max_steps_override, difficulty)
    # Environments with negative rewards use shorter horizon to avoid
    # wasting time on bad policies that run full episodes without terminating.
    all_tasks = [
        ('CartPole-v1', 4, 2, 500, 'normal'),
        ('Acrobot-v1', 6, 3, 150, 'normal'),
        ('MountainCar-v0', 2, 3, 200, 'hard'),     # Sparse reward, needs exploration
        ('LunarLander-v3', 8, 4, 400, 'hard'),     # Complex dynamics, needs diversity
    ]

    # Filter to specific envs if requested
    if args.envs:
        all_tasks = [(name, n, m, ms, d) for name, n, m, ms, d in all_tasks if name in args.envs]

    run_id = time.strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'results_rl', 'Lamarckian')
    os.makedirs(base_output_dir, exist_ok=True)

    print(f"Lamarckian GA-KAN RL Experiments")
    print(f"Output dir: {base_output_dir}")
    print(f"Tasks: {[t[0] for t in all_tasks]}")
    print(f"Config: pop={args.pop_size}, gen={args.max_gen}, "
          f"train_iters={args.n_train_iters}/{args.n_train_iters_elite}(elite), "
          f"envs={args.n_envs}, device={args.device}")

    all_results = []
    for env_name, n_in, m_out, max_steps_env, difficulty in all_tasks:
        # Override max_steps per environment
        args_copy = argparse.Namespace(**vars(args))
        args_copy.max_steps = max_steps_env
        args_copy.task_difficulty = difficulty
        result = run_single_task(env_name, n_in, m_out, args_copy, base_output_dir)
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY - Lamarckian GA-KAN Results")
    print(f"{'='*60}")
    print(f"{'Environment':<20} {'Best Reward':>12} {'Final Avg':>12} {'Time (s)':>10}")
    print(f"{'-'*56}")
    for r in all_results:
        print(f"{r['env_name']:<20} {r['best_reward']:>12.2f} {r['final_avg_reward']:>12.2f} {r['elapsed_seconds']:>10.1f}")

    # Save summary JSON
    summary_path = os.path.join(base_output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'run_id': run_id, 'args': vars(args), 'results': all_results}, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
