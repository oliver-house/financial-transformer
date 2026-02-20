# Financial Time-Series Transformer

A transformer-based model for predicting next-day returns from historical AAPL OHLCV data. Currently contains the data preparation pipeline; model training will follow.

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

## Data

| Setting | Value |
|---------|-------|
| Ticker | AAPL |
| Sequence length | 30 days |
| Features | `ret` (raw), `ret_mean`, `ret_std`, `Volume` (normalised) |
| Target | Next-day return |
| Split | 70% train / 15% val / 15% test |
