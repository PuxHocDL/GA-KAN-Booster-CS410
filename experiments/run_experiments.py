import argparse
import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ga_kan.chromosome import ChromosomeConfig
from ga_kan.genetic_operators import TournamentSelection, GAKANCrossover, BitFlipMutation
from ga_kan.optimizer import GAKANOptimizer
from ga_kan.fitness import build_optimal_model
from experiments.data_loader import (
    get_toy_dataset_1, get_toy_dataset_2, get_toy_dataset_3, get_toy_dataset_4,
    load_uci_classification, load_moons_dataset, load_circles_dataset,
    load_diabetes_regression
)
from experiments.baselines import run_sklearn_baseline, run_standard_kan

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help="Run with small pop/gen for testing")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = args.device
    print(f"Using device: {device}")
    
    datasets = [
        # Classification - sklearn
        ('Iris', lambda: load_uci_classification('iris')),
        ('Wine', lambda: load_uci_classification('wine')),
        ('WDBC', lambda: load_uci_classification('wdbc')),
        ('Digits', lambda: load_uci_classification('digits')),
        # Classification - UCI
        ('Raisin', lambda: load_uci_classification('raisin')),
        ('Rice', lambda: load_uci_classification('rice')),
        ('Banknote', lambda: load_uci_classification('banknote')),
        ('Seeds', lambda: load_uci_classification('seeds')),
        ('Glass', lambda: load_uci_classification('glass')),
        # Classification - synthetic
        ('Moons', load_moons_dataset),
        ('Circles', load_circles_dataset),
        # Regression - toy
        ('Toy1_Eq_6a', get_toy_dataset_1),
        ('Toy2_Eq_6b', get_toy_dataset_2),
        ('Toy3_sincos', get_toy_dataset_3),
        ('Toy4_radial', get_toy_dataset_4),
        # Regression - real
        ('Diabetes', load_diabetes_regression),
    ]
    
    if args.fast:
        pop_size = 10
        max_gen = 5
        N_steps = 5
    else:
        pop_size = 100
        max_gen = 50
        N_steps = 50
        
    results = []
    base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(base_output_dir, exist_ok=True)
    
    for name, loader in datasets:
        dataset_dir = os.path.join(base_output_dir, name.replace(' ', '_').replace('(', '').replace(')', ''))
        os.makedirs(dataset_dir, exist_ok=True)
        
        print(f"\n============================")
        print(f"Dataset: {name}")
        print(f"============================")
        
        # Load Data
        data, task_type = loader()
        if data is None:
            continue
            
        D_train, D_val = data, data
        # Move data to device
        for k in D_train:
            D_train[k] = D_train[k].to(device)
            
        n_features = D_train['train_input'].shape[1]
        
        if task_type == 'classification':
            n_classes = len(torch.unique(D_train['train_label']))
            m_out = n_classes
        else:
            m_out = 1
            
        # 1. Run GA-KAN
        print("Running GA-KAN...")
        config = ChromosomeConfig(n=n_features, m=m_out, d_max=5, u_max=10)
        selection = TournamentSelection()
        crossover = GAKANCrossover(pc=0.9)
        mutation = BitFlipMutation(pm=0.5)
        
        optimizer = GAKANOptimizer(
            config=config,
            selection_strategy=selection,
            crossover_strategy=crossover,
            mutation_strategy=mutation,
            pop_size=pop_size,
            max_gen=max_gen,
            N_steps=N_steps,
            task_type=task_type,
            device=device
        )
        
        best_ind, best_fit = optimizer.run(D_train, D_val)
        
        # Evaluate GA-KAN Best Individual
        best_model = build_optimal_model(best_ind, device=device)
        print("Training final GA-KAN architecture...")
        opt = torch.optim.LBFGS(best_model.parameters(), lr=0.1)
        if task_type == 'classification':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()
            
        for t in range(N_steps):
            def closure():
                opt.zero_grad()
                pred = best_model(D_train['train_input'])
                loss = criterion(pred, D_train['train_label'])
                loss.backward()
                return loss
            opt.step(closure)
            
        with torch.no_grad():
            preds = best_model(D_val['test_input'])
            if task_type == 'classification':
                preds_class = torch.argmax(preds, dim=1)
                gakan_score = accuracy_score(D_val['test_label'].cpu().numpy(), preds_class.cpu().numpy())
            else:
                gakan_score = criterion(preds, D_val['test_label']).item()
                
        # Extract and save interpretability
        print("Extracting and saving Interpretability...")
        optimal_network, feature_scores = optimizer.extract_interpretability(best_model, D_train)
        
        # Save Interpretability Report
        report_path = os.path.join(dataset_dir, 'interpretability_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Dataset: {name}\n")
            f.write(f"Task Type: {task_type}\n")
            f.write(f"Best Fitness (Loss): {best_fit:.4f}\n")
            target_depth, grid_val, _ = best_ind.decode()
            f.write(f"Optimal Depth: {target_depth}\n")
            f.write(f"Optimal Grid Value: {grid_val}\n")
            f.write(f"\nFeature Scores: {feature_scores}\n")
            
            f.write("\nSymbolic Formulas Extracted:\n")
            for l in range(target_depth):
                f.write(f"Layer {l}:\n")
                funs_name = optimal_network.symbolic_fun[l].funs_name
                f.write(str(funs_name) + "\n")
                
        # Save Plot
        try:
            plot_path = os.path.join(dataset_dir, 'ga_kan_architecture.png')
            optimal_network.plot(folder=dataset_dir, title=f"GA-KAN Optimal ({name})")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Architecture plot saved: {plot_path}")
        except Exception as e:
            print(f"Warning: Failed to plot network. {e}")
            try:
                plt.close('all')
            except:
                pass

        # 2. Run Baselines
        print("Running Baselines...")
        # Move back to CPU for sklearn
        D_train_cpu = {k: v.cpu() for k, v in D_train.items()}
        D_val_cpu = {k: v.cpu() for k, v in D_val.items()}
        
        for baseline in ['SVM', 'RF', 'MLP', 'KNN']:
            res = run_sklearn_baseline(baseline, D_train_cpu, D_val_cpu, task_type)
            score = res['Accuracy'] if task_type == 'classification' else res['MSE']
            results.append({
                'Dataset': name,
                'Model': baseline,
                'Score': score
            })
            
        # Standard KAN configs
        # config 1: [d, 2d+1, C]
        res_kan1 = run_standard_kan([n_features, 2*n_features+1, m_out], D_train, D_val, task_type, N_steps, device)
        score1 = res_kan1['Accuracy'] if task_type == 'classification' else res_kan1['MSE']
        results.append({'Dataset': name, 'Model': 'Standard KAN [d, 2d+1, C]', 'Score': score1})
        
        # config 2: [d, 5, 5, 5, C]
        res_kan2 = run_standard_kan([n_features, 5, 5, 5, m_out], D_train, D_val, task_type, N_steps, device)
        score2 = res_kan2['Accuracy'] if task_type == 'classification' else res_kan2['MSE']
        results.append({'Dataset': name, 'Model': 'Standard KAN [d, 5, 5, 5, C]', 'Score': score2})
        
        # Append GA-KAN to results now that it's done
        results.append({
            'Dataset': name,
            'Model': 'GA-KAN',
            'Score': gakan_score
        })
        
        # Save intermediate results
        pd.DataFrame(results).to_csv(os.path.join(base_output_dir, 'results.csv'), index=False)
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(base_output_dir, 'results.csv'), index=False)
    print(f"\nFinal Results saved to {os.path.join(base_output_dir, 'results.csv')}")
    print(df)

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
