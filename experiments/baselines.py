from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from kan import MultKAN as KAN
import torch

def run_sklearn_baseline(model_name, D_train, D_val, task_type='classification'):
    X_train = D_train['train_input'].numpy()
    y_train = D_train['train_label'].numpy().ravel() # sklearn expects 1D target
        
    X_val = D_val['test_input'].numpy()
    y_val = D_val['test_label'].numpy().ravel()
        
    if task_type == 'classification':
        if model_name == 'SVM':
            model = SVC()
        elif model_name == 'RF':
            model = RandomForestClassifier()
        elif model_name == 'MLP':
            # Peer competitor MLP configuration
            model = MLPClassifier(hidden_layer_sizes=(5, 5, 5, 5), max_iter=500)
        elif model_name == 'KNN':
            model = KNeighborsClassifier()
    else:
        if model_name == 'SVM':
            model = SVR()
        elif model_name == 'RF':
            model = RandomForestRegressor()
        elif model_name == 'MLP':
            model = MLPRegressor(hidden_layer_sizes=(5, 5, 5, 5), max_iter=500)
        elif model_name == 'KNN':
            model = KNeighborsRegressor()
            
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    
    if task_type == 'classification':
        score = accuracy_score(y_val, preds)
        metric_name = 'Accuracy'
    else:
        score = mean_squared_error(y_val, preds)
        metric_name = 'MSE'
        
    return {metric_name: score}

def run_standard_kan(width, D_train, D_val, task_type='classification', N_steps=20, device='cpu'):
    model = KAN(width=width, grid=5, k=3, seed=42, device=device, auto_save=False)
    
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)
    
    if task_type == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()
        
    for t in range(N_steps):
        def closure():
            optimizer.zero_grad()
            pred = model(D_train['train_input'])
            train_loss = criterion(pred, D_train['train_label'])
            train_loss.backward()
            return train_loss
        optimizer.step(closure)
        
    # Validation
    with torch.no_grad():
        val_pred = model(D_val['test_input'])
        if task_type == 'classification':
            val_loss = criterion(val_pred, D_val['test_label']).item()
            preds = torch.argmax(val_pred, dim=1)
            acc = accuracy_score(D_val['test_label'].cpu().numpy(), preds.cpu().numpy())
            return {'Loss': val_loss, 'Accuracy': acc}
        else:
            val_loss = criterion(val_pred, D_val['test_label']).item()
            return {'Loss': val_loss, 'MSE': val_loss}
