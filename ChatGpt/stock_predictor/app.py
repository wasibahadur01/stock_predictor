"""Streamlit dashboard for the stock_predictor project.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtesting import run_signal_backtest
from data import fetch_close_prices, fetch_price_data
from features import (
    add_technical_indicators,
    create_model_dataset,
    get_latest_feature_row,
    get_latest_lstm_sequence,
)
from models import MODEL_ORDER, make_metrics_table, predict_tomorrow, train_and_evaluate_all
from portfolio import calculate_returns, optimise_portfolio
from sentiment import fetch_news_sentiment

ASSETS = {
    "^IXIC": "NASDAQ — Nasdaq Composite Index",
    "QQQ": "QQQ — Nasdaq 100 ETF",
    "SPY": "SPY — S&P 500 ETF",
    "GLD": "GLD — Gold ETF",
    "BTC-USD": "BTC-USD — Bitcoin",
}

COMPARISON_SYMBOLS = ("^IXIC", "QQQ", "GLD", "BTC-USD", "SPY")

TRAINING_WINDOWS = {
    "6 months": 183,
    "1 year": 365,
    "2 years": 365 * 2,
}

st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide",
)


def format_currency(value: float, symbol: str) -> str:
    """Format model price output."""

    decimals = 0 if symbol == "BTC-USD" else 2
    return f"${value:,.{decimals}f}"


def format_metric_value(value: float, is_percent: bool = False) -> str:
    """Format metrics for display tables."""

    if pd.isna(value):
        return "—"
    if is_percent:
        return f"{value:.2%}"
    return f"{value:.4f}"


def get_secret_or_env(name: str) -> str:
    """Read a value from Streamlit secrets or environment variables."""

    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, "")


def window_start(end: date, label: str) -> date:
    """Return start date for a sidebar training-window label."""

    return end - timedelta(days=TRAINING_WINDOWS[label])


def show_selected_model_metrics(result) -> None:
    """Display the main metrics for the currently selected model."""

    cols = st.columns(6)
    cols[0].metric("Selected model", result.name)
    cols[1].metric("RMSE", f"{result.metrics['RMSE']:.4f}")
    cols[2].metric("R²", f"{result.metrics['R2']:.4f}")
    cols[3].metric("Accuracy", f"{result.metrics['Accuracy']:.2%}")
    cols[4].metric("Precision", f"{result.metrics['Precision']:.2%}")
    cols[5].metric("Recall", f"{result.metrics['Recall']:.2%}")


def make_metrics_display_table(metrics_table: pd.DataFrame) -> pd.DataFrame:
    """Format model-comparison metrics for a beginner-friendly table."""

    display = metrics_table.copy()
    for column in ["Accuracy", "Precision", "Recall"]:
        display[column] = display[column].apply(lambda value: format_metric_value(value, True))
    for column in ["RMSE", "R²", "Error Std"]:
        display[column] = display[column].apply(lambda value: format_metric_value(value, False))
    return display


def make_price_chart(data: pd.DataFrame, symbol: str, chart_type: str) -> go.Figure:
    """Create historical price chart with Bollinger Bands."""

    indicators = add_technical_indicators(data)
    fig = go.Figure()

    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=indicators.index,
                open=indicators["Open"],
                high=indicators["High"],
                low=indicators["Low"],
                close=indicators["Close"],
                name=symbol,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=indicators.index,
                y=indicators["Close"],
                mode="lines",
                name="Close",
            )
        )

    # Bollinger Bands: upper/lower lines plus shaded area between them.
    fig.add_trace(
        go.Scatter(
            x=indicators.index,
            y=indicators["BB_Upper"],
            mode="lines",
            name="Bollinger Upper",
            line={"width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=indicators.index,
            y=indicators["BB_Lower"],
            mode="lines",
            name="Bollinger Lower",
            fill="tonexty",
            line={"width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=indicators.index,
            y=indicators["BB_Middle"],
            mode="lines",
            name="20-day SMA / BB Middle",
        )
    )

    fig.update_layout(
        title=f"Historical Price with Bollinger Bands — {symbol}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=560,
    )
    return fig


def make_rsi_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """Create RSI chart."""

    indicators = add_technical_indicators(data)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=indicators.index,
            y=indicators["RSI_14"],
            mode="lines",
            name="RSI 14",
        )
    )
    fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought 70")
    fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold 30")
    fig.update_layout(
        title=f"RSI — {symbol}",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis_range=[0, 100],
        hovermode="x unified",
        height=360,
    )
    return fig


def make_macd_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """Create MACD line, signal line, and histogram chart."""

    indicators = add_technical_indicators(data)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=indicators.index,
            y=indicators["MACD_Histogram"],
            name="MACD Histogram",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=indicators.index,
            y=indicators["MACD_Line"],
            mode="lines",
            name="MACD Line",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=indicators.index,
            y=indicators["MACD_Signal"],
            mode="lines",
            name="Signal Line",
        )
    )
    fig.update_layout(
        title=f"MACD — {symbol}",
        xaxis_title="Date",
        yaxis_title="MACD",
        hovermode="x unified",
        height=380,
    )
    return fig


def make_prediction_chart(predictions: pd.DataFrame, symbol: str, model_name: str) -> go.Figure:
    """Create predicted-vs-actual close chart with confidence interval band."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions["Actual_Close"],
            mode="lines",
            name="Actual Close",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions["CI_Upper"],
            mode="lines",
            name="Prediction +1σ error",
            line={"width": 0},
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions["CI_Lower"],
            mode="lines",
            name="Prediction -1σ error",
            fill="tonexty",
            line={"width": 0},
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions["Predicted_Close"],
            mode="lines",
            name="Predicted Close",
        )
    )
    fig.update_layout(
        title=f"Predicted vs Actual Next-Day Close — {symbol} — {model_name}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        height=560,
    )
    return fig


def make_confusion_matrix_chart(cm, model_name: str) -> go.Figure:
    """Create a confusion matrix heatmap."""

    labels = ["Down (0)", "Up (1)"]
    text = [[str(value) for value in row] for row in cm]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Predicted {label}" for label in labels],
            y=[f"Actual {label}" for label in labels],
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
            showscale=True,
        )
    )
    fig.update_layout(
        title=f"Direction Prediction Confusion Matrix — {model_name}",
        xaxis_title="Predicted Direction",
        yaxis_title="Actual Direction",
        height=500,
    )
    return fig


def normalize_price_frame(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Normalize all assets to 100 after every selected asset has a valid price."""

    cleaned = close_prices.sort_index().ffill().dropna(how="any")
    if cleaned.empty:
        raise ValueError("Not enough overlapping data to build the comparison chart.")
    return cleaned / cleaned.iloc[0] * 100


def make_normalized_comparison_chart(close_prices: pd.DataFrame) -> go.Figure:
    """Plot normalized performance for all comparison assets."""

    normalized = normalize_price_frame(close_prices)

    fig = go.Figure()
    for symbol in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[symbol],
                mode="lines",
                name=ASSETS.get(symbol, symbol).split(" — ")[0],
            )
        )

    fig.update_layout(
        title="Normalized Performance Comparison — Starting Value = 100",
        xaxis_title="Date",
        yaxis_title="Normalized performance",
        hovermode="x unified",
        height=560,
    )
    return fig


def make_backtest_chart(backtest: pd.DataFrame, symbol: str, model_name: str) -> go.Figure:
    """Create strategy-vs-buy-and-hold equity curve chart."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["Strategy_Equity"],
            mode="lines",
            name="Model Strategy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["Buy_Hold_Equity"],
            mode="lines",
            name="Buy & Hold",
        )
    )
    fig.update_layout(
        title=f"Backtest Equity Curve — {symbol} — {model_name}",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        height=560,
    )
    return fig


def make_sentiment_gauge(score: float, label: str) -> go.Figure:
    """Create a bullish/bearish sentiment meter."""

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".3f"},
            title={"text": f"News sentiment: {label}"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"thickness": 0.3},
                "steps": [
                    {"range": [-1, -0.05]},
                    {"range": [-0.05, 0.05]},
                    {"range": [0.05, 1]},
                ],
            },
        )
    )
    fig.update_layout(height=320)
    return fig


def make_frontier_chart(frontier: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    """Create efficient frontier chart with max-Sharpe marker."""

    fig = go.Figure()
    if not frontier.empty:
        fig.add_trace(
            go.Scatter(
                x=frontier["Volatility"],
                y=frontier["Return"],
                mode="lines+markers",
                name="Efficient Frontier",
            )
        )

    max_sharpe = summary.loc[summary["Portfolio"] == "Max Sharpe"].iloc[0]
    equal_weight = summary.loc[summary["Portfolio"] == "Equal Weight"].iloc[0]

    fig.add_trace(
        go.Scatter(
            x=[max_sharpe["Annual Volatility"]],
            y=[max_sharpe["Annual Return"]],
            mode="markers",
            marker={"size": 14},
            name="Max Sharpe",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[equal_weight["Annual Volatility"]],
            y=[equal_weight["Annual Return"]],
            mode="markers",
            marker={"size": 12},
            name="Equal Weight",
        )
    )

    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        hovermode="closest",
        height=560,
    )
    return fig


def prediction_csv(predictions: pd.DataFrame, model_name: str) -> bytes:
    """Build downloadable CSV bytes for selected model predictions."""

    export = predictions.copy()
    export.index.name = "Date"
    export.insert(0, "Model", model_name)
    export["Predicted_Label"] = export["Predicted_Direction"].map({1: "UP", 0: "DOWN"})
    export["Actual_Label"] = export["Actual_Direction"].map({1: "UP", 0: "DOWN"})
    return export.to_csv().encode("utf-8")


def build_accuracy_window_table(symbol: str, end_date: date) -> pd.DataFrame:
    """Compare direction accuracy across 6m, 1y, and 2y windows.

    LSTM is intentionally skipped here to keep the comparison responsive. The
    main selected-window run still trains LSTM when enabled.
    """

    rows = []
    for label in TRAINING_WINDOWS:
        try:
            start = window_start(end_date, label)
            data = fetch_price_data(symbol, start, end_date)
            model_data = create_model_dataset(data)
            bundle = train_and_evaluate_all(model_data, include_lstm=False)
            row = {"Training Window": label, "Clean Rows": len(model_data)}
            for model_name in ["Linear Regression", "Random Forest", "XGBoost"]:
                if model_name in bundle.results:
                    row[f"{model_name} Accuracy"] = bundle.results[model_name].metrics[
                        "Accuracy"
                    ]
                else:
                    row[f"{model_name} Accuracy"] = np.nan
            rows.append(row)
        except Exception as exc:
            rows.append({"Training Window": label, "Clean Rows": 0, "Error": str(exc)})

    return pd.DataFrame(rows)


st.title("📈 Stock Predictor")
st.caption(
    "Predict next-day closing price and direction for NASDAQ, QQQ, SPY, GLD, and Bitcoin."
)
st.warning(
    "Learning project only — not financial advice. Markets are risky, and these models "
    "should not be used for real trading decisions."
)

with st.sidebar:
    st.header("Settings")

    selected_symbol = st.selectbox(
        "Asset",
        options=list(ASSETS.keys()),
        format_func=lambda ticker: ASSETS[ticker],
    )

    today = date.today()
    selected_window = st.selectbox(
        "Training data range",
        options=list(TRAINING_WINDOWS.keys()),
        index=1,
        help="Choose how much historical data the models train on.",
    )
    end_date = st.date_input("End date", value=today, max_value=today)
    start_date = window_start(end_date, selected_window)

    chart_type = st.radio("Historical chart type", ["Candlestick", "Line"], horizontal=True)
    requested_model = st.selectbox("Model predictions to display", MODEL_ORDER, index=1)
    include_lstm = st.checkbox("Train LSTM model", value=True, help="PyTorch LSTM is slower.")
    lstm_epochs = st.slider("LSTM epochs", min_value=5, max_value=50, value=20, step=5)

    st.divider()
    st.write("**Run command**")
    st.code("streamlit run app.py", language="bash")

if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()

try:
    with st.spinner("Fetching data, building features, and training models..."):
        price_data = fetch_price_data(selected_symbol, start_date, end_date)
        model_data = create_model_dataset(price_data)
        trained = train_and_evaluate_all(
            model_data,
            include_lstm=include_lstm,
            lstm_epochs=lstm_epochs,
        )
        latest_features = get_latest_feature_row(price_data)
except Exception as exc:
    st.error(str(exc))
    st.stop()

available_models = [model for model in MODEL_ORDER if model in trained.results]
if not available_models:
    st.error("No models could be trained for this selection.")
    st.stop()

if requested_model not in trained.results:
    fallback_model = available_models[0]
    st.sidebar.warning(f"{requested_model} is unavailable. Showing {fallback_model} instead.")
    selected_model = fallback_model
else:
    selected_model = requested_model

selected_result = trained.results[selected_model]
latest_close = float(price_data["Close"].iloc[-1])
latest_date = price_data.index[-1].date()
latest_lstm_sequence = (
    get_latest_lstm_sequence(price_data) if selected_model == "LSTM" else None
)
tomorrow = predict_tomorrow(
    trained,
    latest_features,
    model_data,
    selected_model,
    latest_close=latest_close,
    latest_lstm_sequence=latest_lstm_sequence,
)
metrics_table = make_metrics_table(trained)

test_rows = len(selected_result.predictions)

with st.sidebar:
    st.divider()
    st.write("**Dataset summary**")
    st.write(f"Selected range: `{start_date}` to `{end_date}`")
    st.write(f"Rows downloaded: `{len(price_data):,}`")
    st.write(f"Clean model rows: `{len(model_data):,}`")
    st.write(f"Selected test rows: `{test_rows:,}`")
    st.write(f"Latest data date: `{latest_date}`")
    if trained.unavailable:
        with st.expander("Unavailable model notes"):
            for model_name, reason in trained.unavailable.items():
                st.write(f"**{model_name}:** {reason}")

summary_cols = st.columns(5)
summary_cols[0].metric("Selected asset", selected_symbol)
summary_cols[1].metric("Training window", selected_window)
summary_cols[2].metric("Latest close", format_currency(latest_close, selected_symbol))
summary_cols[3].metric("Clean model rows", f"{len(model_data):,}")
summary_cols[4].metric("Test split", "Last 20%")

(
    historical_tab,
    prediction_tab,
    direction_tab,
    tomorrow_tab,
    comparison_tab,
    backtest_tab,
    portfolio_tab,
) = st.tabs(
    [
        "1. Historical + Indicators",
        "2. Predicted vs Actual",
        "3. Direction + Models",
        "4. Tomorrow + News",
        "5. Multi-Asset Comparison",
        "6. Backtest",
        "7. Portfolio Optimiser",
    ]
)

with historical_tab:
    st.subheader("Historical price chart with Bollinger Bands")
    show_selected_model_metrics(selected_result)
    st.plotly_chart(
        make_price_chart(price_data, selected_symbol, chart_type),
        use_container_width=True,
    )
    st.subheader("RSI")
    st.plotly_chart(make_rsi_chart(price_data, selected_symbol), use_container_width=True)
    st.subheader("MACD")
    st.plotly_chart(make_macd_chart(price_data, selected_symbol), use_container_width=True)
    with st.expander("View recent OHLCV + indicator data"):
        st.dataframe(add_technical_indicators(price_data).tail(50), use_container_width=True)

with prediction_tab:
    st.subheader("Predicted vs actual next-day closing price")
    show_selected_model_metrics(selected_result)
    st.info(
        "The split is time-based: first 80% for training and last 20% for testing. "
        "No shuffling is used. The shaded band is ±1 standard deviation of historical prediction error."
    )
    st.plotly_chart(
        make_prediction_chart(selected_result.predictions, selected_symbol, selected_model),
        use_container_width=True,
    )
    st.download_button(
        "Download selected model predictions as CSV",
        data=prediction_csv(selected_result.predictions, selected_model),
        file_name=f"{selected_symbol.replace('^', '')}_{selected_model.replace(' ', '_')}_predictions.csv",
        mime="text/csv",
    )
    with st.expander("View prediction table"):
        st.dataframe(selected_result.predictions.tail(100), use_container_width=True)

with direction_tab:
    st.subheader("Direction prediction accuracy and model comparison")
    show_selected_model_metrics(selected_result)
    st.plotly_chart(
        make_confusion_matrix_chart(selected_result.confusion_matrix, selected_model),
        use_container_width=True,
    )
    st.caption("Direction labels: 1 = next close is higher than current close; 0 = next close is lower or equal.")

    st.subheader("Side-by-side model metrics")
    st.dataframe(make_metrics_display_table(metrics_table), use_container_width=True)

    st.subheader("How direction accuracy changes by training window")
    with st.spinner("Training quick comparison models for 6m, 1y, and 2y windows..."):
        window_accuracy = build_accuracy_window_table(selected_symbol, end_date)
    formatted_accuracy = window_accuracy.copy()
    for column in formatted_accuracy.columns:
        if "Accuracy" in column:
            formatted_accuracy[column] = formatted_accuracy[column].apply(
                lambda value: format_metric_value(value, True)
            )
    st.dataframe(formatted_accuracy, use_container_width=True)
    st.caption("This window table skips LSTM to keep the dashboard responsive; LSTM metrics are shown in the main comparison table when trained.")

with tomorrow_tab:
    st.subheader("Tomorrow / next available data-day prediction")
    show_selected_model_metrics(selected_result)

    direction_text = "UP 📈" if tomorrow["predicted_direction"] == 1 else "DOWN 📉"
    tomorrow_cols = st.columns(4)
    tomorrow_cols[0].metric(
        "Predicted close",
        format_currency(float(tomorrow["predicted_price"]), selected_symbol),
    )
    tomorrow_cols[1].metric("Predicted direction", direction_text)
    tomorrow_cols[2].metric("Direction confidence", f"{float(tomorrow['confidence']):.2%}")
    tomorrow_cols[3].metric("±1σ price error", format_currency(float(tomorrow["error_std"]), selected_symbol))

    st.write(
        f"Latest close on `{latest_date}` was **{format_currency(latest_close, selected_symbol)}**. "
        f"Model shown: **{selected_model}**."
    )

    st.subheader("News sentiment")
    newsapi_key = get_secret_or_env("NEWSAPI_KEY")
    if not newsapi_key:
        st.info(
            "News sentiment is ready, but NEWSAPI_KEY is missing. Add it to Streamlit secrets "
            "or as an environment variable to fetch the 10 latest headlines."
        )
    else:
        try:
            sentiment_result = fetch_news_sentiment(selected_symbol, newsapi_key, page_size=10)
            st.plotly_chart(
                make_sentiment_gauge(sentiment_result.average_score, sentiment_result.label),
                use_container_width=True,
            )
            st.dataframe(sentiment_result.headlines, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not fetch news sentiment: {exc}")

with comparison_tab:
    st.subheader("NASDAQ, QQQ, Gold, Bitcoin, and S&P 500 comparison")
    show_selected_model_metrics(selected_result)
    try:
        comparison_prices = fetch_close_prices(COMPARISON_SYMBOLS, start_date, end_date)
        st.plotly_chart(
            make_normalized_comparison_chart(comparison_prices),
            use_container_width=True,
        )
        with st.expander("View normalized comparison data"):
            normalized = normalize_price_frame(comparison_prices)
            st.dataframe(normalized.tail(50), use_container_width=True)
    except Exception as exc:
        st.error(str(exc))

with backtest_tab:
    st.subheader("Simple model-signal backtest")
    show_selected_model_metrics(selected_result)
    initial_capital = st.number_input(
        "Initial capital ($)", min_value=100.0, value=10_000.0, step=500.0
    )
    transaction_cost_pct = st.number_input(
        "Transaction cost per buy/sell (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.10,
        step=0.05,
    )
    try:
        backtest = run_signal_backtest(
            selected_result.predictions,
            initial_capital=initial_capital,
            transaction_cost_pct=transaction_cost_pct / 100,
        )
        st.plotly_chart(
            make_backtest_chart(backtest, selected_symbol, selected_model),
            use_container_width=True,
        )
        bcols = st.columns(4)
        bcols[0].metric("Strategy final value", f"${backtest['Strategy_Equity'].iloc[-1]:,.2f}")
        bcols[1].metric("Buy-hold final value", f"${backtest['Buy_Hold_Equity'].iloc[-1]:,.2f}")
        bcols[2].metric("Strategy P/L", f"${backtest['Strategy_PnL'].iloc[-1]:,.2f}")
        bcols[3].metric("Buy-hold P/L", f"${backtest['Buy_Hold_PnL'].iloc[-1]:,.2f}")
        with st.expander("View backtest table"):
            st.dataframe(backtest.tail(100), use_container_width=True)
    except Exception as exc:
        st.error(str(exc))

with portfolio_tab:
    st.subheader("Portfolio optimiser using Modern Portfolio Theory")
    st.caption(
        "Select 2 to 5 assets. The optimiser uses historical daily returns, long-only weights, "
        "and maximises the Sharpe ratio."
    )
    portfolio_assets = st.multiselect(
        "Portfolio assets",
        options=list(ASSETS.keys()),
        default=["QQQ", "SPY", "GLD", "BTC-USD"],
        format_func=lambda ticker: ASSETS[ticker],
    )
    risk_free_rate = st.number_input(
        "Annual risk-free rate (%)",
        min_value=0.0,
        max_value=25.0,
        value=2.0,
        step=0.25,
    )

    if len(portfolio_assets) < 2 or len(portfolio_assets) > 5:
        st.info("Choose between 2 and 5 assets to run the optimiser.")
    else:
        try:
            portfolio_prices = fetch_close_prices(tuple(portfolio_assets), start_date, end_date)
            returns = calculate_returns(portfolio_prices)
            optimisation = optimise_portfolio(returns, risk_free_rate=risk_free_rate / 100)

            weights = optimisation["weights"].copy()
            summary = optimisation["summary"].copy()
            frontier = optimisation["frontier"]

            st.plotly_chart(make_frontier_chart(frontier, summary), use_container_width=True)

            weights_display = weights.copy()
            weights_display["Optimal Weight"] = weights_display["Optimal Weight"].map(lambda x: f"{x:.2%}")
            weights_display["Equal Weight"] = weights_display["Equal Weight"].map(lambda x: f"{x:.2%}")
            st.subheader("Suggested allocation")
            st.dataframe(weights_display, use_container_width=True)

            summary_display = summary.copy()
            for column in ["Annual Return", "Annual Volatility"]:
                summary_display[column] = summary_display[column].map(lambda x: f"{x:.2%}")
            summary_display["Sharpe Ratio"] = summary_display["Sharpe Ratio"].map(lambda x: f"{x:.3f}")
            st.subheader("Portfolio summary")
            st.dataframe(summary_display, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))
