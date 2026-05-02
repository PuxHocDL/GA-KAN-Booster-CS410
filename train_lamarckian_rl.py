import os
import yaml
import time
import logging
import torch

from ga_kan.chromosome import ChromosomeConfig
from ga_kan.genetic_operators import TournamentSelection, UniformCrossover, BitFlipMutation
from ga_kan.lamarck_optimizer import LamarckGAKANOptimizer
from ga_kan.fitness import build_optimal_model

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("LamarckianRL")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'history.txt'))
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def main():
    # 1. Config Environment & Run Identifier
    env_name = 'CartPole-v1'  # Default environment
    run_id = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("experiments", f"lamarck_{env_name}_{run_id}")
    
    logger = setup_logger(experiment_dir)
    logger.info(f"Starting Lamarckian Evolution on {env_name}")
    logger.info(f"Results will be saved in: {experiment_dir}")

    # 2. Config GA-KAN
    # Giả sử obs_dim = 4, action_dim = 2 cho CartPole
    n = 4
    m = 2
    d_max = 3
    u_max = 8
    
    config = ChromosomeConfig(n=n, m=m, d_max=d_max, u_max=u_max)
    
    selection = TournamentSelection(tournament_size=3)
    crossover = UniformCrossover(crossover_rate=0.7)
    mutation = BitFlipMutation(pm=0.03)  # Lower mutation rate to preserve good topologies
    
    pop_size = 6  # Smaller pop, more training per individual
    max_gen = 15
    n_steps = 8  # Vectorized envs
    n_train_iterations = 5  # New individuals: architecture screening
    n_train_iterations_elite = 20  # Elite: heavy training (fine-tuning inherited weights)
    max_steps_per_iter = 500  # Full CartPole episode length
    device = 'cpu'  # Force CPU for VM
    
    logger.info(f"Config: pop_size={pop_size}, max_gen={max_gen}, device={device}")
    
    # Save config
    config_dict = {
        'env_name': env_name,
        'n': n,
        'm': m,
        'd_max': d_max,
        'u_max': u_max,
        'pop_size': pop_size,
        'max_gen': max_gen,
        'n_steps': n_steps,
        'n_train_iterations': n_train_iterations,
        'n_train_iterations_elite': n_train_iterations_elite,
        'max_steps_per_iter': max_steps_per_iter,
        'device': device,
        'type': 'lamarckian_evolution',
        'note': 'No NormalizeObservation - shared env avoids stat corruption'
    }
    with open(os.path.join(experiment_dir, 'config.yml'), 'w') as f:
        yaml.dump(config_dict, f)

    # 3. Init & Run Optimizer
    optimizer = LamarckGAKANOptimizer(
        config=config,
        selection_strategy=selection,
        crossover_strategy=crossover,
        mutation_strategy=mutation,
        pop_size=pop_size,
        max_gen=max_gen,
        N_steps=n_steps,
        device=device,
        n_train_iterations=n_train_iterations,
        n_train_iterations_elite=n_train_iterations_elite,
        max_steps_per_iter=max_steps_per_iter,
        vectorization_mode='sync',
        dense_init=True,
        num_workers=pop_size  # Parallelize across all individuals
    )
    
    best_ind, best_fitness = optimizer.run(env_name=env_name, logger=logger)
    
    # 4. Save Results
    logger.info("=== Evolution Finished ===")
    logger.info(f"Best Target Depth: {best_ind.decode()[0]}")
    logger.info(f"Best Fitness (Max Reward): {-best_fitness:.4f}")
    
    # Extract final model and save state
    final_model = build_optimal_model(best_ind, device=device)
    if best_ind.has_weights():
        final_model.load_state_dict(best_ind.weights)
        
    save_path = os.path.join(experiment_dir, 'best_model.pth')
    torch.save({
        'bits': best_ind.bits.tolist(),
        'state_dict': final_model.state_dict(),
        'fitness': best_fitness,
        'config': config_dict
    }, save_path)
    
    logger.info(f"Best model state and chromosome bits saved to {save_path}")

if __name__ == "__main__":
    main()
