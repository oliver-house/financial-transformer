"""Train and evaluate classical baseline models on the same data splits as the transformer."""

import json
import os
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from data_prep import TICKERS, SEQ_LEN

DATA_PATH = 'data/splits.npz'
RUNS_DIR  = 'runs'
SEED      = 42

def load_splits(path=DATA_PATH):
    data = np.load(path)
    X_train, y_train = data['X_train'], data['y_train']
    X_test,  y_test  = data['X_test'],  data['y_test']
    # Flatten (N, seq_len, n_features) -> (N, seq_len * n_features) for tabular models
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat  = X_test.reshape(len(X_test),  -1)
    return X_train_flat, y_train, X_test_flat, y_test

def compute_metrics(preds, targets):
    preds, targets = np.array(preds), np.array(targets)
    mse     = float(np.mean((preds - targets) ** 2))
    mae     = float(np.mean(np.abs(preds - targets)))
    dir_acc = float(np.mean((preds * targets) > 0))
    return {'MSE': mse, 'MAE': mae, 'DirAcc': dir_acc}

def print_report(label, metrics):
    print(
        f"  {label:<30s}  "
        f"MSE={metrics['MSE']:.6f}  "
        f"MAE={metrics['MAE']:.6f}  "
        f"DirAcc={metrics['DirAcc']:.4f}"
    )

def main():
    X_train, y_train, X_test, y_test = load_splits()
    print(f'Train: {X_train.shape}, Test: {X_test.shape}\n')

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest':     RandomForestRegressor(n_estimators=50, max_depth=8, random_state=SEED, n_jobs=-1),
        'XGBoost':           XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                          subsample=0.8, random_state=SEED, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        print(f'Training {name} ...')
        model.fit(X_train, y_train)
        preds   = model.predict(X_test)
        metrics = compute_metrics(preds, y_test)
        print_report(name, metrics)
        results[name] = metrics

    print()
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, 'baseline_metrics.json')
    payload = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'tickers':   TICKERS,
        'seq_len':   SEQ_LEN,
        'metrics':   results,
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'Baseline metrics saved to {path}')

if __name__ == '__main__':
    main()
