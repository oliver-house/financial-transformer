# Financial Time-Series Transformer

A transformer-based model for predicting next-day returns from historical OHLCV data, compared against classical ML baselines.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Prepare data**
```bash
python data_prep.py
```
Downloads daily OHLCV data for 10 large-cap US equities via `yfinance`, computes return-based features, normalises each stock independently using train-set statistics, and saves concatenated train/val/test splits to `data/splits.npz`.

**Train transformer**
```bash
python train.py
```
Trains a `TinyTransformer` on the prepared splits, evaluates against zero and persistence baselines, and saves artefacts to `runs/`: `metrics.json` and `preds_vs_true.png`.

**Train classical baselines**
```bash
python baselines.py
```
Trains Linear Regression, Random Forest, and XGBoost on the same flattened feature windows and saves results to `runs/baseline_metrics.json`.

**Compare all models**
```bash
python compare.py
```
Prints a unified comparison table of all models across MSE, MAE, and directional accuracy.

## Data

| Setting | Value |
|---------|-------|
| Tickers | AAPL, MSFT, JPM, JNJ, XOM, WMT, CAT, GS, NEE, AMZN |
| Sequence length | 30 days |
| Features | `ret` (raw), `ret_mean`, `ret_std`, `Volume` (normalised) |
| Target | Next-day return |
| Split | 70% train / 15% val / 15% test (per stock, then concatenated) |

## Model

| Setting | Value |
|---------|-------|
| Architecture | TinyTransformer (single encoder layer) |
| Pooling | Last token |
| Optimiser | AdamW, lr=1e-3 |
| Batch size | 256 |
| Max epochs | 50 (early stopping, patience=5) |

## Results

| Model | MSE | MAE | DirAcc |
|-------|-----|-----|--------|
| Zero baseline | 0.000311 | 0.01197 | 0.00% |
| Persistence baseline | 0.000652 | 0.01728 | 50.15% |
| Linear Regression | 0.000313 | 0.01202 | 51.23% |
| XGBoost | 0.000313 | 0.01202 | 51.70% |
| Transformer | 0.000314 | 0.01204 | 52.60% |
| Random Forest | 0.000314 | 0.01197 | 52.84% |
