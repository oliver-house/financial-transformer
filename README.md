# Financial Time-Series Machine Learning Project

The Python script `data_prep.py` prepares financial time-series data as rolling sequences suitable for machine learning models. The current implementation focuses on constructing a clean, validated dataset from historical data.

The script downloads daily OHLCV data using `yfinance` and computes simple return-based features:

- Daily return
- 5-day rolling mean of returns
- 5-day rolling standard deviation of returns

It constructs rolling input sequences of fixed length, defines prediction targets as next-day returns, and performs structural validation checks on the resulting datasets. 

This repository currently contains only the data preparation pipeline.