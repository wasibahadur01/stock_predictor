# 📈 Stock Price Predictor — Full ML Dashboard

An end-to-end machine learning dashboard for predicting stock prices and market direction across 5 major assets. Built with Python, scikit-learn, XGBoost, and Streamlit.

**[▶ Live Demo](https://your-app-name.streamlit.app)** ← replace after deploying

---

## Features

| Feature | Detail |
|---|---|
| 4 ML Models | Linear Regression, XGBoost, NumPy LSTM, Random Forest |
| 9 Technical Indicators | RSI, MACD, Bollinger Bands, MA5/20/50, Volatility, Momentum |
| Backtesting | Strategy vs Buy & Hold P&L curve with Sharpe ratio |
| Portfolio Optimiser | Efficient frontier via Monte Carlo + scipy (max Sharpe) |
| News Sentiment | TextBlob NLP on live headlines (NewsAPI) or sample data |
| Confidence Intervals | ±1σ shaded band on all price prediction charts |
| Model Comparison | Side-by-side RMSE / R² / MAPE table for all 3 price models |
| 5 Assets | NASDAQ, QQQ, Gold, Bitcoin, S&P 500 |
| Date Range Picker | 6 months / 1 year / 2 years of training data |
| Multi-asset Chart | Normalized performance comparison |
| CSV Download | Export all predictions with confidence intervals |
| Telegram Alerts | Daily alert when confidence > 70% |
| Live Data Toggle | Real Yahoo Finance data via yfinance |

---

## Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/stock-predictor.git
cd stock-predictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download TextBlob corpus (one time)
python -c "import nltk; nltk.download('punkt')"

# 5. Run the dashboard
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Deploy to Streamlit Cloud (Free — 10 minutes)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: stock price predictor"
git remote add origin https://github.com/YOUR_USERNAME/stock-predictor.git
git push -u origin main
```

### Step 2 — Sign up for Streamlit Cloud

Go to **share.streamlit.io** → Sign in with GitHub (free account).

### Step 3 — Deploy

1. Click **"New app"**
2. Select your GitHub repo
3. Set **Main file path** to: `app.py`
4. Click **Deploy**

Your app will be live at:
```
https://YOUR_USERNAME-stock-predictor-app-XXXXX.streamlit.app
```

### Step 4 — Add NewsAPI key as a secret (optional)

In Streamlit Cloud → your app → **Settings → Secrets**:
```toml
NEWSAPI_KEY = "your_key_from_newsapi.org"
```

Then in `app.py` sidebar, the key input will auto-populate from `st.secrets`.

### Troubleshooting deployment

| Error | Fix |
|---|---|
| `ModuleNotFoundError` | Check `requirements.txt` has the package |
| `textblob punkt not found` | Add `python -c "import nltk; nltk.download('punkt')"` to a `packages.txt` file or use `st.cache_resource` |
| App crashes on startup | Check Streamlit Cloud logs under your app → **Manage app → Logs** |
| Memory error | Reduce `n_simulations` in `optimise_portfolio()` from 3000 to 1000 |

---

## Telegram Alerts Setup

### Step 1 — Create a bot
1. Open Telegram → search **@BotFather**
2. Send `/newbot` → follow prompts → copy the **BOT_TOKEN**

### Step 2 — Get your Chat ID
1. Send any message to your new bot
2. Visit: `https://api.telegram.org/bot<BOT_TOKEN>/getUpdates`
3. Copy the `"id"` value from `"chat"` in the response

### Step 3 — Set environment variables
```bash
# Windows
set TELEGRAM_BOT_TOKEN=your_token_here
set TELEGRAM_CHAT_ID=your_chat_id_here

# Mac/Linux
export TELEGRAM_BOT_TOKEN=your_token_here
export TELEGRAM_CHAT_ID=your_chat_id_here
```

### Step 4 — Run manually or schedule
```bash
# Run once
python alerts.py

# Schedule daily at 8am (Linux cron)
crontab -e
# Add this line:
0 8 * * 1-5 /path/to/venv/bin/python /path/to/stock-predictor/alerts.py >> /tmp/alerts.log 2>&1
```

---

## Project Structure

```
stock_predictor/
├── app.py              # Streamlit dashboard (5 tabs, 860 lines)
├── model.py            # Data, features, all 4 ML models, backtest, portfolio
├── alerts.py           # Telegram alert system
├── requirements.txt    # Clean dependencies for deployment
├── .streamlit/
│   └── config.toml     # Dark theme + server config
└── README.md
```

---

## Dashboard Tabs

### 📊 Predictions
- Next-day price prediction with confidence interval
- Model comparison table (LR vs XGBoost vs LSTM)
- Price chart with ±1σ shaded band
- Actual vs predicted scatter plot
- Direction confidence scatter
- Feature importance (Random Forest)
- Candlestick + Bollinger Bands
- RSI + MACD charts
- CSV download (all 3 models + CI)

### ⚡ Backtest
- ML strategy P&L curve vs buy & hold
- Strategy/B&H ratio chart
- Sharpe ratio, max drawdown, number of trades
- Model comparison across all 3 price models

### 🌍 Compare Assets
- Normalized performance chart (all 5 assets)
- Performance summary table (return, volatility, Sharpe)
- Training period accuracy comparison

### 💼 Portfolio Optimiser
- Select 2–5 assets
- Monte Carlo efficient frontier (3,000 simulations)
- Max-Sharpe allocation via scipy optimisation
- Interactive donut chart for allocation
- Equal-weight baseline comparison

### 📰 Sentiment
- Overall bullish/bearish sentiment meter
- Per-headline polarity + subjectivity scores
- Polarity bar chart
- Powered by TextBlob NLP (+ NewsAPI for live headlines)

---

## ML Models Explained

| Model | Predicts | How |
|---|---|---|
| Linear Regression | Next-day price | Learns linear relationship between 9 features and price |
| XGBoost | Next-day price | Gradient boosted trees — better at nonlinear patterns |
| NumPy LSTM | Next-day price | Sequence model — looks at last 60 days of prices |
| Random Forest | UP or DOWN | Ensemble of 200 decision trees on 9 features |

**Why NumPy LSTM instead of PyTorch?**
PyTorch CPU wheel is ~900MB — too large for free deploy environments (Streamlit Cloud has a 1GB limit). This NumPy implementation does the same LSTM forward pass math and produces equivalent predictions. On your local machine, swap it for `torch.nn.LSTM` for GPU support and faster training.

---

## What I Learned

- Building a full ML pipeline from raw price data to deployed web app
- Feature engineering: RSI, MACD, Bollinger Bands, moving averages from scratch
- Why you can't shuffle time-series data (train/test split must preserve order)
- XGBoost vs Linear Regression — gradient boosting for financial data
- Implementing LSTM forward pass in pure NumPy (sigmoid, tanh, cell/hidden state)
- Backtesting a trading strategy and interpreting Sharpe ratio and drawdown
- Modern Portfolio Theory — efficient frontier, Sharpe ratio maximisation
- NLP sentiment analysis with TextBlob on financial headlines
- Deploying a multi-tab Streamlit app to Streamlit Cloud
- Telegram Bot API for automated alerts

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Models predicting stock direction achieve ~50–55% accuracy on real data — only slightly above random chance. Do not use this for real investment decisions.

---

Built by [Wasi](https://github.com/YOUR_USERNAME) · Python · scikit-learn · XGBoost · Streamlit · Plotly · TextBlob · scipy
