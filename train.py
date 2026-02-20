"""Train and evaluate the TinyTransformer on financial return data."""

import json
import os
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')          # headless — no display required
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from model import TinyTransformer

DATA_PATH   = 'data/splits.npz'
RUNS_DIR    = 'runs'
TICKER      = 'AAPL'
SEED        = 42
BATCH_SIZE  = 256
LR          = 1e-3
EPOCHS      = 50
PATIENCE    = 5          # early-stopping patience (set to EPOCHS to disable)
IDX_RET     = 0          # index of 'ret' in FEAT_COLS
PLOT_N      = 200        # number of test points shown in the line plot

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ReturnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def compute_metrics(preds, targets):
    """Return MSE, MAE, and directional accuracy."""
    mse = nn.functional.mse_loss(preds, targets).item()
    mae = (preds - targets).abs().mean().item()
    dir_acc = ((preds * targets) > 0).float().mean().item()
    return {'MSE': mse, 'MAE': mae, 'DirAcc': dir_acc}

def print_report(label, metrics):
    print(
        f"  {label:<20s}  "
        f"MSE={metrics['MSE']:.6f}  "
        f"MAE={metrics['MAE']:.6f}  "
        f"DirAcc={metrics['DirAcc']:.4f}"
    )

def evaluate_baselines(X_test, y_test):
    """Return metrics dict for both baselines and print them."""
    y = torch.from_numpy(y_test)

    zero_preds    = torch.zeros_like(y)
    persist_preds = torch.from_numpy(X_test[:, -1, IDX_RET])

    zero_metrics    = compute_metrics(zero_preds, y)
    persist_metrics = compute_metrics(persist_preds, y)

    print_report('Zero baseline',        zero_metrics)
    print_report('Persistence baseline', persist_metrics)

    return {'zero': zero_metrics, 'persistence': persist_metrics}

def save_metrics(
    run_dir,
    ticker,
    seq_len,
    model_cfg,
    baseline_metrics,
    transformer_metrics,
):
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'ticker':    ticker,
        'seq_len':   seq_len,
        'config': {
            'batch_size': BATCH_SIZE,
            'lr':         LR,
            'epochs':     EPOCHS,
            'patience':   PATIENCE,
            'seed':       SEED,
            **model_cfg,
        },
        'metrics': {
            'zero_baseline':        baseline_metrics['zero'],
            'persistence_baseline': baseline_metrics['persistence'],
            'transformer':          transformer_metrics,
        },
    }
    path = os.path.join(run_dir, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'Metrics saved to {path}')


def save_plot(run_dir, true, pred, n):
    os.makedirs(run_dir, exist_ok=True)
    true, pred = true[:n], pred[:n]
    xs = np.arange(len(true))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # Top: line plot — true vs predicted over time
    ax = axes[0]
    ax.plot(xs, true, label='True return', linewidth=0.9, alpha=0.85)
    ax.plot(xs, pred, label='Predicted return', linewidth=0.9, alpha=0.85, linestyle='--')
    ax.set_title(f'True vs Predicted returns (first {len(true)} test samples)')
    ax.set_xlabel('Test sample index')
    ax.set_ylabel('Normalised return')
    ax.legend()
    ax.grid(True, linewidth=0.4)

    # Bottom: scatter — pred vs true
    ax = axes[1]
    ax.scatter(true, pred, s=6, alpha=0.4)
    lim = max(np.abs(true).max(), np.abs(pred).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=0.8, label='y = x')
    ax.set_title('Scatter: Predicted vs True')
    ax.set_xlabel('True return')
    ax.set_ylabel('Predicted return')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend()
    ax.grid(True, linewidth=0.4)

    fig.tight_layout()
    path = os.path.join(run_dir, 'preds_vs_true.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'Plot saved to {path}')

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Return (mean_loss, all_preds, all_targets)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        total_loss += criterion(preds, y_batch).item() * len(y_batch)
        all_preds.append(preds.cpu())
        all_targets.append(y_batch.cpu())
    mean_loss = total_loss / len(loader.dataset)
    return mean_loss, torch.cat(all_preds), torch.cat(all_targets)

def main():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    data = np.load(DATA_PATH)
    X_train, y_train = data['X_train'], data['y_train']
    X_val,   y_val   = data['X_val'],   data['y_val']
    X_test,  y_test  = data['X_test'],  data['y_test']
    print(f'Splits — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}')

    # DataLoaders
    train_loader = DataLoader(ReturnDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ReturnDataset(X_val,   y_val),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(ReturnDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

    # Model
    d_features = X_train.shape[2]
    seq_len    = X_train.shape[1]
    model_cfg  = dict(d_features=d_features, seq_len=seq_len, d_model=32, n_heads=4, dropout=0.1, pool='last')
    model = TinyTransformer(**model_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Training
    print(f'\nTraining for up to {EPOCHS} epochs (patience={PATIENCE}) ...\n')
    best_val_loss = float('inf')
    best_state    = None
    patience_ctr  = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        improved = val_loss < best_val_loss
        marker = ' *' if improved else ''
        print(f'Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.6f}  val={val_loss:.6f}{marker}')

        if improved:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f'\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).')
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluation report
    print('\n--- Test-set evaluation ---')
    baseline_metrics = evaluate_baselines(X_test, y_test)
    _, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    transformer_metrics = compute_metrics(test_preds, test_targets)
    print_report('Transformer', transformer_metrics)

    # Save artefacts
    print()
    save_metrics(
        run_dir=RUNS_DIR,
        ticker=TICKER,
        seq_len=seq_len,
        model_cfg=model_cfg,
        baseline_metrics=baseline_metrics,
        transformer_metrics=transformer_metrics,
    )
    save_plot(
        run_dir=RUNS_DIR,
        true=test_targets.numpy(),
        pred=test_preds.numpy(),
    )

if __name__ == '__main__':
    main()