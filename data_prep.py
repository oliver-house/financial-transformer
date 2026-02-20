"""Prepare financial time-series data as rolling sequences for machine learning."""

import os
import numpy as np
import yfinance as yf

SEQ_LEN = 30
DATA_PATH = 'data/splits.npz'
TICKER = 'AAPL'
PERIOD = 'max'
INTERVAL = '1d'
WINDOW_SIZE = 5
OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
FEAT_COLS = ['ret', 'ret_mean', 'ret_std', 'Volume']
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

def load_data(
    ticker=TICKER, 
    period=PERIOD, 
    interval=INTERVAL, 
    window_size=WINDOW_SIZE, 
    ohlcv_cols=OHLCV_COLS
    ):
    """Download OHLCV data and compute return-based features."""
    try:
        df = yf.download(ticker, period=period, interval=interval)
    except Exception as e:
        raise RuntimeError(f'Failed to download data for ticker "{ticker}": {e}') from e
    if df.empty:
        raise ValueError(f'No data returned for ticker "{ticker}" with period="{period}", interval="{interval}"')
    df = df[ohlcv_cols]
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
    """Main function to prepare and validate data."""
    df = load_data()
    X, y = prepare_data(df)
    N = X.shape[0]
    n_train = int(TRAIN_FRAC * N)
    n_val = int(VAL_FRAC * N)
    n_test = N - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f'Bad split sizes: N={N}, n_train={n_train}, n_val={n_val}, n_test={n_test}')
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]
    eps = 1e-8
    norm_means, norm_stds = [], []
    norm_cols = [c for c in FEAT_COLS if c != 'ret']  
    for col in norm_cols:
        idx = FEAT_COLS.index(col)
        m = X_train[:, :, idx].mean()
        s = X_train[:, :, idx].std()
        norm_means.append(m)
        norm_stds.append(s)
        for X_split in (X_train, X_val, X_test):
            X_split[:, :, idx] = (X_split[:, :, idx] - m) / (s + eps)

    print('Split sizes:', X_train.shape[0], X_val.shape[0], X_test.shape[0])
    dir_name = os.path.dirname(DATA_PATH)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    np.savez(
        DATA_PATH,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        norm_means=np.array(norm_means),
        norm_stds=np.array(norm_stds),
    )
    print(f'Splits saved to {DATA_PATH}')
    validate_data(df, X, y)

if __name__ == '__main__':
    main()
