import argparse
import sys
import os
import torch
import copy
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing as mp
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ga_kan.chromosome import ChromosomeConfig
from ga_kan.genetic_operators import TournamentSelection, GAKANCrossover, BitFlipMutation
from ga_kan.optimizer import GAKANOptimizer
from ga_kan.fitness import build_optimal_model
from ga_kan.fitness_rl import evaluate_fitness_rl, train_rl_vectorized


def _cpu_worker_eval(args):
    """
    Process-pool worker: evaluate a single chromosome on CPU with its own
    small vectorized env. Each process pins torch to 1 thread so that
    `pop_size` processes do not oversubscribe the cores.
    """
    individual, env_name, n_inner_envs = args
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    return evaluate_fitness_rl(individual, env_name, N_steps=n_inner_envs, device='cpu', vectorization_mode='sync')


class GAKANOptimizerRL(GAKANOptimizer):
    def __init__(self, *args, num_workers=None, n_inner_envs=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers
        self.n_inner_envs = n_inner_envs

    def run(self, env_name: str):
        """
        RL Specific Pipeline (Vòng lặp GA cho Gymnasium).

        On CPU we parallelise ACROSS the population (each individual is
        independent) using a ProcessPoolExecutor. Each worker runs its own
        small SyncVectorEnv. This is the right model when you have many
        CPU cores and a small/cheap policy network.
        """
        self.initialize_population()

        use_pool = self.device == 'cpu' and (self.num_workers is None or self.num_workers > 1)
        executor = None
        if use_pool:
            ctx = mp.get_context('spawn')
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=ctx,
            )

        # On GPU path we still reuse one AsyncVectorEnv across the whole run.
        shared_envs = None
        if not use_pool:
            try:
                shared_envs = gym.make_vec(env_name, num_envs=self.N_steps, vectorization_mode="async")
            except TypeError:
                shared_envs = gym.make_vec(env_name, num_envs=self.N_steps)

        pbar = tqdm(range(self.max_gen), desc=f"GA Optimization ({env_name})", unit="gen")
        try:
            for gen in pbar:
                if use_pool:
                    payload = [(ind, env_name, self.n_inner_envs) for ind in self.population]
                    fitnesses = list(executor.map(_cpu_worker_eval, payload, chunksize=1))
                else:
                    fitnesses = [
                        evaluate_fitness_rl(ind, env_name, self.N_steps, self.device, envs=shared_envs)
                        for ind in self.population
                    ]

                for ind, fit in zip(self.population, fitnesses):
                    if fit < self.best_fitness:
                        self.best_fitness = fit
                        self.best_individual = copy.deepcopy(ind)

                avg_fit = sum(fitnesses) / len(fitnesses)
                pbar.set_postfix({'Best Reward': f'{-self.best_fitness:.2f}', 'Avg Reward': f'{-avg_fit:.2f}'})

                # Selection & Crossover & Mutation (Same as parent)
                new_population = []
                if self.best_individual is not None:
                    new_population.append(copy.deepcopy(self.best_individual))

                while len(new_population) < self.pop_size:
                    parents = self.selection_strategy.select(self.population, fitnesses, num_parents=2)
                    p1, p2 = parents[0], parents[1]
                    c1, c2 = self.crossover_strategy.crossover(p1, p2)
                    self.mutation_strategy.mutate(c1)
                    self.mutation_strategy.mutate(c2)
                    new_population.append(c1)
                    if len(new_population) < self.pop_size:
                        new_population.append(c2)

                self.population = new_population
        finally:
            if shared_envs is not None:
                shared_envs.close()
            if executor is not None:
                executor.shutdown(wait=True)

        return self.best_individual, self.best_fitness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Process pool size for CPU mode. Defaults to os.cpu_count().')
    parser.add_argument('--n-inner-envs', type=int, default=4,
                        help='num_envs inside each per-individual SyncVectorEnv (CPU mode).')
    parser.add_argument('--pop-size', type=int, default=80)
    parser.add_argument('--max-gen', type=int, default=20)
    parser.add_argument('--n-steps', type=int, default=50,
                        help='num_envs for the shared AsyncVectorEnv (GPU mode only).')
    args = parser.parse_args()

    device = args.device
    cpu_count = os.cpu_count() or 1
    num_workers = args.num_workers if args.num_workers is not None else cpu_count
    print(f"Using device: {device}")
    if device == 'cpu':
        print(f"CPU mode: {num_workers} worker processes x {args.n_inner_envs} inner envs each "
              f"(host has {cpu_count} cores)")

    pop_size = args.pop_size
    max_gen = args.max_gen
    N_steps = args.n_steps
    
    rl_tasks = [
        ('CartPole-v1', 4, 2),
        ('Acrobot-v1', 6, 3),
        ('MountainCar-v0', 2, 3),
        ('LunarLander-v3', 8, 4)
    ]
    
    base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_rl')
    os.makedirs(base_output_dir, exist_ok=True)
    
    for env_name, n_in, m_out in rl_tasks:
        print(f"\n============================")
        print(f"RL Environment: {env_name}")
        print(f"============================")
        
        dataset_dir = os.path.join(base_output_dir, env_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        config = ChromosomeConfig(n=n_in, m=m_out, d_max=5, u_max=10)
        selection = TournamentSelection(tournament_size=2)
        crossover = GAKANCrossover(pc=0.8)
        mutation = BitFlipMutation(pm=0.2)
        
        optimizer = GAKANOptimizerRL(
            config=config,
            selection_strategy=selection,
            crossover_strategy=crossover,
            mutation_strategy=mutation,
            pop_size=pop_size,
            max_gen=max_gen,
            N_steps=N_steps,
            device=device,
            num_workers=num_workers,
            n_inner_envs=args.n_inner_envs,
        )
        
        best_ind, best_fit = optimizer.run(env_name)
        
        # Training the final GA-KAN policy to print its performance
        best_model = build_optimal_model(best_ind, device=device)
        print("Training final GA-KAN policy using Vectorized REINFORCE...")
        
        opt = torch.optim.Adam(best_model.parameters(), lr=0.01)
        train_envs = gym.make_vec(env_name, num_envs=10)
        
        final_rewards = []
        for ep in range(5): # 5 opt steps on batches of 10 episodes
            reward = train_rl_vectorized(best_model, train_envs, opt, device=device)
            final_rewards.append(reward)
            
        print(f"Final Trained Policy - Avg Reward (over concurrent episodes): {sum(final_rewards)/len(final_rewards):.2f}")
        train_envs.close()
        
        # Save Plot
        try:
            # Fix PyKAN NaN variance issue from batch=1 steps:
            # We push a dummy batch of states to calculate stable activations for plotting
            dummy_states = torch.randn(100, n_in, device=device)
            _ = best_model(dummy_states)
            
            best_model.plot(folder=dataset_dir, title=f"GA-KAN Policy ({env_name})")
            plot_path = os.path.join(dataset_dir, 'optimal_policy_network.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Policy plot saved in {dataset_dir}")
        except Exception as e:
            print(f"Warning: Failed to plot policy network. {e}")
            
        # Record Video of the trained policy
        print(f"Recording gameplay video for {env_name}...")
        try:
            record_env = gym.make(env_name, render_mode="rgb_array")
            record_env = gym.wrappers.RecordVideo(record_env, video_folder=dataset_dir, name_prefix=f"{env_name}_gameplay", episode_trigger=lambda x: True)
            
            state, _ = record_env.reset()
            done = False
            truncated = False
            while not (done or truncated):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits = best_model(state_tensor)
                    # Deterministic action for evaluation
                    action = torch.argmax(logits, dim=1).item() 
                state, reward, done, truncated, _ = record_env.step(action)
            record_env.close()
            print(f"Gameplay video successfully saved in {dataset_dir}")
        except Exception as e:
            print(f"Failed to record video: {e}")

    # --- Auto Zip & Push to Hugging Face ---
    try:
        import zipfile
        from huggingface_hub import HfApi
        
        print("\nZipping results...")
        experiments_dir = os.path.dirname(os.path.abspath(__file__))
        zip_path = os.path.join(experiments_dir, "results.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, strict_timestamps=False) as zipf:
            for folder in ['results', 'results_rl']:
                folder_path = os.path.join(experiments_dir, folder)
                if os.path.exists(folder_path):
                    for root, dirs, files in os.walk(folder_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, experiments_dir)
                            zipf.write(file_path, arcname)
                            
        print("Pushing to Hugging Face (PuxAI/CS410)...")
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("[Error] HF_TOKEN not found in environment variables.")
        else:
            api = HfApi(token=hf_token)
            api.create_repo(repo_id="PuxAI/CS410", repo_type="dataset", exist_ok=True)
            api.upload_file(
                path_or_fileobj=zip_path,
                path_in_repo="results.zip",
                repo_id="PuxAI/CS410",
                repo_type="dataset"
            )
            print("Successfully pushed results.zip to Hugging Face!")
    except ImportError:
        print("\n[Warning] huggingface_hub is not installed. Skipping auto-push. (Run 'pip install huggingface_hub' to enable)")
    except Exception as e:
        print(f"\n[Error] Failed to push to Hugging Face: {e}")

if __name__ == '__main__':
    main()
