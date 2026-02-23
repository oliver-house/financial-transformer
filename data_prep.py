"""Prepare financial time-series data as rolling sequences for machine learning."""

import os
import numpy as np
import pandas as pd
import yfinance as yf

SEQ_LEN = 30
DATA_PATH = 'data/splits.npz'
TICKERS = ['AAPL', 'MSFT', 'JPM', 'JNJ', 'XOM', 'WMT', 'CAT', 'GS', 'NEE', 'AMZN']
PERIOD = 'max'
INTERVAL = '1d'
WINDOW_SIZE = 5
OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
FEAT_COLS = ['ret', 'ret_mean', 'ret_std', 'Volume']
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

def load_data(
    ticker,
    period=PERIOD,
    interval=INTERVAL,
    window_size=WINDOW_SIZE,
    ohlcv_cols=OHLCV_COLS
    ):
    """Download OHLCV data and compute return-based features for one ticker."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    except Exception as e:
        raise RuntimeError(f'Failed to download data for ticker "{ticker}": {e}') from e
    if df.empty:
        raise ValueError(f'No data returned for ticker "{ticker}" with period="{period}", interval="{interval}"')
    df = df[ohlcv_cols]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['ret'] = df['Close'].pct_change()
    df['ret_mean'] = df['ret'].rolling(window_size).mean()
    df['ret_std']  = df['ret'].rolling(window_size).std()
    df = df.dropna()
    return df

def prepare_data(df, seq_len=SEQ_LEN, feat_cols=FEAT_COLS):
    """Prepare rolling sequences and target returns."""
    X_list, y_list = [], []
    for i in range(seq_len - 1, len(df) - 1):
        X_window = df.iloc[i - seq_len + 1 : i + 1][feat_cols].to_numpy(dtype=np.float32)
        y_target = np.float32(df.iloc[i + 1]['ret'])
        X_list.append(X_window)
        y_list.append(y_target)
    return np.array(X_list), np.array(y_list)

def validate_data(df, X, y, seq_len=SEQ_LEN, feat_cols=FEAT_COLS):
    """Validate structural consistency of prepared datasets."""
    if not (X.ndim == 3 and y.ndim == 1):
        raise ValueError(f'Expected X.ndim=3 and y.ndim=1, got X.ndim={X.ndim}, y.ndim={y.ndim}')
    if X.shape[0] != y.shape[0]:
        raise ValueError(f'X and y sample count mismatch: X.shape={X.shape}, y.shape={y.shape}')
    if X.shape[1] != seq_len:
        raise ValueError(f'X sequence length mismatch: X.shape={X.shape}, expected seq_len={seq_len}')
    if X.shape[2] != len(feat_cols):
        raise ValueError(f'X feature count mismatch: X.shape={X.shape}, expected {len(feat_cols)} features')
    if not np.isfinite(X).all():
        raise ValueError('X contains NaN or inf values')
    if not np.isfinite(y).all():
        raise ValueError('y contains NaN or inf values')
    last_i = seq_len + X.shape[0] - 2
    last_input_end_date = df.index[last_i]
    last_target_date = df.index[last_i + 1]
    if last_target_date <= last_input_end_date:
        raise ValueError(f'Temporal ordering violation: last input end {last_input_end_date} >= last target {last_target_date}')

def main():
    """Download, process, and split data for all tickers, then concatenate splits."""
    trains, vals, tests = [], [], []

    for ticker in TICKERS:
        print(f'Processing {ticker} ...')
        df = load_data(ticker)
        X, y = prepare_data(df)
        validate_data(df, X, y)

        N = X.shape[0]
        n_train = int(TRAIN_FRAC * N)
        n_val = int(VAL_FRAC * N)
        if min(n_train, n_val, N - n_train - n_val) <= 0:
            raise ValueError(f'Bad split sizes for {ticker}: N={N}')

        X_train, y_train = X[:n_train], y[:n_train]
        X_val,   y_val   = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
        X_test,  y_test  = X[n_train + n_val :], y[n_train + n_val :]

        # Normalise per-stock using only train-set statistics to avoid leakage.
        # ret is left in raw form: it is already near zero-mean and unit-scale.
        # Rolling mean, std, and volume have very different magnitudes and benefit
        # from standardisation.
        norm_cols = [c for c in FEAT_COLS if c != 'ret']
        eps = 1e-8
        for col in norm_cols:
            idx = FEAT_COLS.index(col)
            m = X_train[:, :, idx].mean()
            s = X_train[:, :, idx].std()
            for X_split in (X_train, X_val, X_test):
                X_split[:, :, idx] = (X_split[:, :, idx] - m) / (s + eps)

        trains.append((X_train, y_train))
        vals.append((X_val, y_val))
        tests.append((X_test, y_test))

    X_train = np.concatenate([t[0] for t in trains])
    y_train = np.concatenate([t[1] for t in trains])
    X_val   = np.concatenate([t[0] for t in vals])
    y_val   = np.concatenate([t[1] for t in vals])
    X_test  = np.concatenate([t[0] for t in tests])
    y_test  = np.concatenate([t[1] for t in tests])

    print(f'\nSplit sizes — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}')

    dir_name = os.path.dirname(DATA_PATH)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    np.savez(
        DATA_PATH,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
    )
    print(f'Splits saved to {DATA_PATH}')

if __name__ == '__main__':
    main()
