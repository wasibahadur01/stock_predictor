# Stock Predictor - Learning Project

A Streamlit dashboard that uses machine learning to predict the next day's closing price and direction for selected assets.

**Disclaimer:** This application is for educational purposes only. It does not constitute financial advice. Past performance does not guarantee future results. Do not use it for real trading decisions.

## Supported Assets
- QQQ (Nasdaq ETF)
- SPY (S&P 500 ETF)
- GLD (Gold ETF)
- BTC-USD (Bitcoin)

## Features
- Fetch historical OHLCV data using yfinance
- Technical indicators as model features
- Regression (Linear Regression) to predict next day close price
- Classification (Random Forest) to predict price direction (Up/Down)
- Time‑based train/test split (no shuffling)
- Evaluation metrics: RMSE, R², Accuracy, Precision, Recall, Confusion Matrix
- Tomorrow's prediction with confidence

## Setup
1. Clone the repository or create the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt