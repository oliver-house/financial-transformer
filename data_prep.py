"""Prepare financial time-series data as rolling sequences for machine learning."""

import numpy as np
import yfinance as yf

SEQ_LEN = 30

def load_data(ticker='AAPL', period='max', interval='1d'):
    """Download OHLCV data and compute return-based features."""
    df = yf.download(ticker, period=period, interval=interval)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = df.columns.get_level_values(0)
    df['ret'] = df['Close'].pct_change()
    df['ret_mean_5'] = df['ret'].rolling(5).mean()
    df['ret_std_5'] = df['ret'].rolling(5).std()
    df = df.dropna()
    return df

def prepare_data(df, seq_len, feat_cols):
    """Prepare rolling sequences and target returns."""
    X_list, y_list = [], []
    for i in range(seq_len - 1, len(df) - 1):
        X_window = df.iloc[i - seq_len + 1 : i + 1][feat_cols].to_numpy(dtype=np.float32)
        y_target = np.float32(df.iloc[i + 1]['ret'])
        X_list.append(X_window)
        y_list.append(y_target)
    return np.array(X_list), np.array(y_list)

def validate_data(df, X, y, seq_len, feat_cols):
    """Validate structural consistency of prepared datasets."""
    assert X.ndim == 3 and y.ndim == 1, (X.ndim, y.ndim)
    assert X.shape[0] == y.shape[0], (X.shape, y.shape)
    assert X.shape[1] == seq_len, (X.shape, seq_len)
    assert X.shape[2] == len(feat_cols), (X.shape, len(feat_cols))
    assert np.isfinite(X).all(), 'X has NaN/inf'
    assert np.isfinite(y).all(), 'y has NaN/inf'
    last_i = seq_len + X.shape[0] - 2
    last_input_end_date = df.index[last_i]
    last_target_date = df.index[last_i + 1]
    assert last_target_date > last_input_end_date, (last_input_end_date, last_target_date)

def main():
    """Main function to prepare and validate data."""
    df = load_data()
    feat_cols = ['ret', 'ret_mean_5', 'ret_std_5', 'Volume']
    X, y = prepare_data(df, SEQ_LEN, feat_cols)
    validate_data(df, X, y, SEQ_LEN, feat_cols)

if __name__ == '__main__':
    main()