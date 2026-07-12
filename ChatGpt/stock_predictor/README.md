# stock_predictor

A beginner-friendly Streamlit machine learning dashboard that predicts:

1. The next available data-day closing price.
2. The next available data-day price direction.
3. A simple model-based trading signal.

> ⚠️ **Learning project only. Not financial advice.** Markets are risky. These models are simple educational examples and should not be used for real trading decisions.

---

## Supported assets

The dashboard supports these Yahoo Finance symbols:

- `^IXIC` — NASDAQ Composite Index
- `QQQ` — Nasdaq 100 ETF
- `SPY` — S&P 500 ETF
- `GLD` — Gold ETF
- `BTC-USD` — Bitcoin

---

## Project structure

```text
stock_predictor/
├── app.py                    # Streamlit dashboard main entry point
├── data.py                   # yfinance data fetching and Streamlit caching
├── features.py               # Technical indicators and model dataset creation
├── models.py                 # Linear, Random Forest, XGBoost, and LSTM models
├── backtesting.py            # Simple buy/sell signal backtest
├── portfolio.py              # Modern Portfolio Theory optimiser
├── sentiment.py              # NewsAPI + TextBlob sentiment scoring
├── alert_bot.py              # Daily Telegram alert script
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml           # Streamlit Cloud theme/server config
```

---

## Main dashboard features

### 1. Date range picker

The sidebar lets the user train on:

- 6 months
- 1 year
- 2 years

The dashboard also shows how direction accuracy changes across those training windows.

### 2. Historical price chart

The dashboard includes:

- Candlestick or line chart
- Bollinger Bands
- 20-day moving average
- Shaded band between the upper and lower Bollinger Bands

### 3. Technical indicators

The model input features are:

- 20-day Simple Moving Average (`SMA_20`)
- 50-day Simple Moving Average (`SMA_50`)
- 14-day Relative Strength Index (`RSI_14`)
- 10-day price momentum percentage (`Momentum_10`)
- Volume change percentage (`Volume_Change_Pct`)
- Daily return percentage (`Daily_Return_Pct`)
- Distance from 20-day SMA as a percentage (`Distance_SMA20_Pct`)

Extra dashboard indicators:

- RSI chart
- MACD line
- MACD signal line
- MACD histogram
- Bollinger Bands

Rows with missing values caused by indicator calculations are dropped before training.

The volume-change feature is defensive: if a symbol has missing or zero volume, the app fills that feature with `0` instead of losing the whole dataset. This is useful for index-style Yahoo Finance symbols.

### 4. Models

The app trains and compares:

| Model | Price prediction | Direction prediction |
|---|---|---|
| Linear Regression | Linear Regression | Direction inferred from predicted price vs current close |
| Random Forest | Random Forest Regressor | Random Forest Classifier |
| XGBoost | XGBoost Regressor | XGBoost Classifier |
| LSTM | PyTorch LSTM on 60-day sequences | Direction inferred from predicted price vs current close |

The split is time-based:

- First 80% of clean rows = training set
- Last 20% of clean rows = test set
- No shuffling

Prediction rows are indexed by the actual next available market date, not by the feature date. This keeps the predicted-vs-actual chart and CSV export easier to understand.

### 5. Metrics

Regression metrics:

- RMSE
- R² score

Classification metrics:

- Accuracy
- Precision
- Recall
- Confusion matrix

The model comparison tab shows all available models side by side.

### 6. Prediction chart with confidence interval

The predicted-vs-actual tab shows:

- Actual closing price
- Predicted closing price
- A shaded confidence band of ±1 standard deviation of historical prediction error

### 7. CSV download

The dashboard includes a CSV download button for the selected stock and selected model.

The exported CSV includes:

- Model name
- Actual close
- Predicted close
- Actual direction
- Predicted direction
- Prediction error
- Confidence interval columns
- Direction labels

### 8. Multi-stock comparison

A separate tab plots normalized performance for:

- NASDAQ Composite
- QQQ
- Gold ETF
- Bitcoin
- S&P 500 ETF

Each line starts at 100 so the user can compare relative performance side by side.

### 9. Backtesting

The backtest tab simulates a simple strategy:

- Buy / stay invested when the model predicts `UP`
- Sell / stay in cash when the model predicts `DOWN`

It compares the model strategy against buy-and-hold on the same chart.

This is intentionally basic and does not fully model real trading conditions.

### 10. News sentiment

The Tomorrow tab can fetch the 10 latest headlines for the selected asset using NewsAPI and score them with TextBlob.

It shows:

- Headline table
- Average sentiment score
- Bullish / Neutral / Bearish meter

You need a free NewsAPI key for this feature.

### 11. Portfolio optimiser

The portfolio tab lets the user select 2 to 5 assets and uses Modern Portfolio Theory to:

- Estimate annualised return and volatility
- Calculate the efficient frontier
- Suggest the long-only allocation that maximises the Sharpe ratio
- Compare the optimal allocation with an equal-weight portfolio

### 12. Daily Telegram alerts

The separate `alert_bot.py` script trains the Random Forest model for all 5 assets and sends a Telegram message when an asset has a predicted direction confidence above 70%.

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

Core packages are kept in `requirements.txt` only when the app actually imports them. `requests` is needed for NewsAPI and Telegram alerts.

---

## Run the dashboard

```bash
streamlit run app.py
```

---

## NewsAPI setup

Create a free API key at NewsAPI.

For local development, create this file:

```text
.streamlit/secrets.toml
```

Add:

```toml
NEWSAPI_KEY = "your_newsapi_key_here"
```

Do not commit real API keys to GitHub.

You can also use an environment variable:

```bash
export NEWSAPI_KEY="your_newsapi_key_here"
```

Windows PowerShell:

```powershell
$env:NEWSAPI_KEY="your_newsapi_key_here"
```

---

## Telegram alert setup

Create a Telegram bot using BotFather and get your bot token.

Set these environment variables:

```bash
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

Optional settings:

```bash
export ALERT_CONFIDENCE_THRESHOLD="0.70"
export ALERT_LOOKBACK_DAYS="730"
```

Run:

```bash
python alert_bot.py
```

You can schedule this script using cron, Windows Task Scheduler, GitHub Actions, or a small cloud server.

Example cron job for every weekday at 8:00 AM:

```cron
0 8 * * 1-5 cd /path/to/stock_predictor && /path/to/python alert_bot.py
```

---

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload all files in the `stock_predictor` folder.
3. Make sure these files are included:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - all helper modules
4. Go to Streamlit Community Cloud.
5. Choose **New app**.
6. Select your GitHub repository.
7. Set the main file path to:

```text
app.py
```

8. Open **Advanced settings**.
9. Add your secret if you want news sentiment:

```toml
NEWSAPI_KEY = "your_newsapi_key_here"
```

10. Deploy the app.

If the app is slow on free hosting, reduce LSTM epochs in the sidebar or uncheck **Train LSTM model**.


---

## Local quality check

Before pushing to GitHub, run a syntax check from the project folder:

```bash
python -m py_compile app.py data.py features.py models.py backtesting.py portfolio.py sentiment.py alert_bot.py
```

The project was also tested with synthetic OHLCV data for:

- indicator creation
- 6-month-style short datasets
- zero-volume data
- one-class direction labels
- all four model families
- tomorrow prediction
- CSV-ready prediction tables
- backtesting
- portfolio optimisation

---

## Notes and limitations

- `yfinance` does not need an API key, but it still depends on Yahoo Finance availability.
- ETF symbols do not trade on weekends or market holidays.
- `BTC-USD` trades daily, so its calendar differs from ETF/index symbols.
- The LSTM model is intentionally small so it can run inside a Streamlit app.
- The backtest is educational and simplified.
- The portfolio optimiser uses historical returns; historical returns do not guarantee future returns.
- The news sentiment score is based on headline text and is not a professional sentiment model.

