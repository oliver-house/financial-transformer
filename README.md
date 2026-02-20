# Financial Time-Series Transformer

A transformer-based model for predicting next-day returns from historical AAPL OHLCV data.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Prepare data**
```bash
python data_prep.py
```
Downloads daily AAPL OHLCV data via `yfinance`, computes return-based features, normalises features using train-set statistics, and saves train/val/test splits to `data/splits.npz`.

**Train model**
```bash
python train.py
```
Trains a `TinyTransformer` on the prepared splits, evaluates against zero and persistence baselines, and saves artefacts to `runs/`: `metrics.json` and `preds_vs_true.png`.

## Data

| Setting | Value |
|---------|-------|
| Ticker | AAPL |
| Sequence length | 30 days |
| Features | `ret` (raw), `ret_mean`, `ret_std`, `Volume` (normalised) |
| Target | Next-day return |
| Split | 70% train / 15% val / 15% test |

## Model

| Setting | Value |
|---------|-------|
| Architecture | Single-layer transformer encoder |
| `d_model` | 32 |
| `n_heads` | 4 |
| Pooling | Last token |
| Optimiser | AdamW, lr=1e-3 |
| Batch size | 256 |
| Max epochs | 50 (early stopping, patience=5) |
