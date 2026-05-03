import torch
import numpy as np
from kan import MultKAN as KAN # PyKAN typically uses KAN, but file is MultKAN
import warnings

from .chromosome import Chromosome, is_valid_topology

def evaluate_fitness(individual: Chromosome, D_train, D_val, task_type='regression', N_steps=20, device='cpu', use_adam=True):
    """
    Evaluates the fitness of an individual chromosome.
    Fitness is the minimum validation loss achieved during training.
    
    use_adam: If True, use Adam (faster for GA search phase). If False, use LBFGS (for final model).
    """
    # 1. Decode
    target_depth, grid_value, active_masks = individual.decode()
    
    # 2. Filtering mechanism (Topology Check)
    if not is_valid_topology(active_masks):
        return float('inf')
        
    # Determine the actual width based on active masks
    # active_masks is a list of arrays: mask[0] is L0->L1, mask[1] is L1->L2 ...
    # Width should be [in_dim, l1_dim, ..., out_dim]
    width = [active_masks[0].shape[0]]
    for mask in active_masks:
        width.append(mask.shape[1])
        
    try:
        # 3. Build Model
        model = KAN(width=width, grid=grid_value, k=3, seed=42, device=device, auto_save=False)
        
        # Inject the decoded topology masks into the model
        for l in range(target_depth):
            mask_tensor = torch.tensor(active_masks[l], dtype=torch.float32, device=device)
            model.act_fun[l].mask.data = mask_tensor
            
    except Exception as e:
        warnings.warn(f"Failed to initialize model with width {width}: {e}")
        return float('inf')
        
    # 4. Train Model
    min_loss = float('inf')
    
    def train():
        nonlocal min_loss
        # Select loss function
        if task_type == 'classification':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()
        
        if use_adam:
            # Adam: much faster per step, good enough for topology search
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            for t in range(N_steps):
                optimizer.zero_grad()
                pred = model(D_train['train_input'])
                train_loss = criterion(pred, D_train['train_label'])
                train_loss.backward()
                optimizer.step()
                
                # Validation every 5 steps to save time
                if t % 5 == 0 or t == N_steps - 1:
                    with torch.no_grad():
                        val_pred = model(D_val['test_input'])
                        val_loss = criterion(val_pred, D_val['test_label']).item()
                        if val_loss < min_loss:
                            min_loss = val_loss
        else:
            # LBFGS: more accurate but slower, used for final training
            optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)
            for t in range(N_steps):
                def closure():
                    optimizer.zero_grad()
                    pred = model(D_train['train_input'])
                    train_loss = criterion(pred, D_train['train_label'])
                    train_loss.backward()
                    return train_loss
                    
                optimizer.step(closure)
                
                with torch.no_grad():
                    val_pred = model(D_val['test_input'])
                    val_loss = criterion(val_pred, D_val['test_label']).item()
                    if val_loss < min_loss:
                        min_loss = val_loss
                    
    try:
        train()
    except Exception as e:
        warnings.warn(f"Training failed: {e}")
        return float('inf')
        
    return min_loss

def build_optimal_model(individual: Chromosome, device='cpu'):
    """
    Helper to just build the PyKAN model for the best individual.
    """
    target_depth, grid_value, active_masks = individual.decode()
    width = [active_masks[0].shape[0]]
    for mask in active_masks:
        width.append(mask.shape[1])
        
    model = KAN(width=width, grid=grid_value, k=3, seed=42, device=device, auto_save=False)
    for l in range(target_depth):
        mask_tensor = torch.tensor(active_masks[l], dtype=torch.float32, device=device)
        model.act_fun[l].mask.data = mask_tensor
        
    return model
