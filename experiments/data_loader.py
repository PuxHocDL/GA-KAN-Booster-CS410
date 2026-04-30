import torch
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings

def normalize_features(X):
    """Normalize features to [-1, 1] for KAN."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(X)

def get_toy_dataset_1(n_samples=1000):
    """Regression dataset for f(x,y) = exp(sin(pi*x) + y^2)"""
    np.random.seed(42)
    # Sample x, y uniformly from [-1, 1]
    X = np.random.uniform(-1, 1, size=(n_samples, 2))
    x, y = X[:, 0], X[:, 1]
    f_xy = np.exp(np.sin(np.pi * x) + y**2)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(f_xy, dtype=torch.float32).unsqueeze(1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    return {'train_input': X_train, 'train_label': y_train, 'test_input': X_test, 'test_label': y_test}, 'regression'

def get_toy_dataset_2(n_samples=1000):
    """Regression dataset for f(x,y) = xy"""
    np.random.seed(42)
    X = np.random.uniform(-1, 1, size=(n_samples, 2))
    f_xy = X[:, 0] * X[:, 1]
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(f_xy, dtype=torch.float32).unsqueeze(1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    return {'train_input': X_train, 'train_label': y_train, 'test_input': X_test, 'test_label': y_test}, 'regression'

def load_uci_classification(dataset_name):
    """Load Iris, Wine, WDBC from sklearn, and Raisin, Rice from ucimlrepo."""
    if dataset_name == 'iris':
        data = load_iris()
        X, y = data.data, data.target
    elif dataset_name == 'wine':
        data = load_wine()
        X, y = data.data, data.target
    elif dataset_name == 'wdbc':
        data = load_breast_cancer()
        X, y = data.data, data.target
    elif dataset_name == 'raisin':
        try:
            from ucimlrepo import fetch_ucirepo
            raisin = fetch_ucirepo(id=850)
            X = raisin.data.features.values
            # Target is string: 'Kecimen', 'Besni'
            y_str = raisin.data.targets.values.ravel()
            y = np.where(y_str == 'Kecimen', 0, 1)
        except Exception as e:
            warnings.warn(f"Failed to load Raisin: {e}")
            return None, None
    elif dataset_name == 'rice':
        try:
            from ucimlrepo import fetch_ucirepo
            rice = fetch_ucirepo(id=545)
            X = rice.data.features.values
            y_str = rice.data.targets.values.ravel()
            y = np.where(y_str == 'Cammeo', 0, 1)
        except Exception as e:
            warnings.warn(f"Failed to load Rice: {e}")
            return None, None
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
        
    X_scaled = normalize_features(X)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long) # Use long for classification labels
    
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    return {'train_input': X_train, 'train_label': y_train, 'test_input': X_test, 'test_label': y_test}, 'classification'
