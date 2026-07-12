"""
app.py — Full Streamlit dashboard for the Stock Price Predictor.

Features:
  ✅ Date range picker (6 months / 1 year / 2 years)
  ✅ Multi-stock normalized comparison chart
  ✅ CSV download of predictions
  ✅ MACD indicator chart
  ✅ Bollinger Bands on candlestick chart
  ✅ Backtesting — strategy vs buy-and-hold P&L curve
  ✅ XGBoost as 3rd model + model comparison table
  ✅ News sentiment meter (NewsAPI + TextBlob)
  ✅ Confidence intervals on price prediction chart
  ✅ NumPy LSTM as 4th model
  ✅ Portfolio optimiser (efficient frontier, max-Sharpe)

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model import (
    TICKERS, PERIOD_DAYS, load_data, load_all_normalized,
    train_models, predict_next_day, run_backtest,
    fetch_sentiment, optimise_portfolio,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stock Predictor")
    st.caption("ML-powered · scikit-learn · XGBoost · LSTM")
    st.divider()

    ticker_name = st.selectbox("Select Asset", list(TICKERS.keys()))

    period_label = st.radio(
        "Training Data Range",
        options=list(PERIOD_DAYS.keys()),
        index=2,
        horizontal=True,
    )
    n_days = PERIOD_DAYS[period_label]

    model_choice = st.selectbox(
        "Price Prediction Model",
        ["Linear Regression", "XGBoost", "LSTM"],
        help="Select which model drives the price prediction and backtest.",
    )

    use_live = st.toggle("Use Live Data (yfinance)", value=False)

    news_api_key = st.text_input(
        "NewsAPI Key (optional)",
        value="",
        type="password",
        help="Free key from newsapi.org — leave blank to use sample headlines.",
    )

    st.divider()
    st.markdown("**Models:**")
    st.markdown("- 🔵 Linear Regression")
    st.markdown("- 🟠 XGBoost")
    st.markdown("- 🟣 LSTM (NumPy)")
    st.markdown("- 🟢 Random Forest (direction)")
    st.divider()
    st.caption("⚠️ Educational only. Not financial advice.")


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_results(ticker_name, use_live, n_days):
    df, is_live = load_data(ticker_name, use_live, n_days=n_days)
    r = train_models(df)
    r["is_live"] = is_live
    return r

@st.cache_data(show_spinner=False)
def get_comparison(use_live, n_days):
    return load_all_normalized(use_live=use_live, n_days=n_days)

@st.cache_data(show_spinner=False)
def get_sentiment(ticker_name, api_key):
    return fetch_sentiment(ticker_name, api_key=api_key)

@st.cache_data(show_spinner=False)
def get_portfolio(selected, use_live, n_days):
    return optimise_portfolio(list(selected), use_live=use_live, n_days=n_days)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Training models for {ticker_name} ({period_label})…"):
    results  = get_results(ticker_name, use_live, n_days)
    comp_df  = get_comparison(use_live, n_days)

prediction = predict_next_day(results, model_choice)
bt         = run_backtest(results, model_choice)
sentiment  = get_sentiment(ticker_name, news_api_key)
pm         = results["price_metrics"]
dm         = results["dir_metrics"]

# ── Data badge ────────────────────────────────────────────────────────────────
if results["is_live"]:
    st.success("✅ Live data from Yahoo Finance")
else:
    st.info("🔬 Synthetic data (realistic simulation). Toggle 'Use Live Data' for real prices.")

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Predictions",
    "⚡ Backtest",
    "🌍 Compare Assets",
    "💼 Portfolio",
    "📰 Sentiment",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title(f"📊 {ticker_name} — {model_choice} Predictions")
    st.caption(
        f"Period: **{period_label}** · Train: {results['train_size']} days · "
        f"Last date: {prediction['last_date']}"
    )

    # ── Prediction cards ──────────────────────────────────────────────────────
    st.subheader("🔮 Next Trading Day Prediction")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Current Price", f"${prediction['current_price']:,.2f}")
    with c2:
        st.metric(
            f"{model_choice} Prediction",
            f"${prediction['predicted_price']:,.2f}",
            delta=f"{prediction['change_pct']:+.2f}%",
        )
    with c3:
        st.metric(
            "Confidence Interval",
            f"±${prediction['resid_std']:,.2f}",
            help="±1 std dev of model's historical prediction errors",
        )
    with c4:
        color = "🟢" if "UP" in prediction["direction"] else "🔴"
        st.metric("Direction (RF)", f"{color} {prediction['direction']}")
    with c5:
        st.metric("RF Confidence", f"{prediction['confidence']*100:.1f}%")

    # Sentiment mini-card inline
    sent_color = {"Bullish 🟢": "green", "Bearish 🔴": "red"}.get(sentiment["overall"], "grey")
    st.markdown(
        f"**News Sentiment:** :{sent_color}[{sentiment['overall']}]  "
        f"(score: {sentiment['avg_score']:+.3f})"
    )
    st.divider()

    # ── Model comparison table ────────────────────────────────────────────────
    st.subheader("🏆 Model Comparison — Price Prediction")
    cmp_rows = [
        {
            "Model":      "🔵 Linear Regression",
            "RMSE ($)":   round(results["lr_metrics"]["rmse"],   2),
            "MAE ($)":    round(results["lr_metrics"]["mae"],    2),
            "R²":         round(results["lr_metrics"]["r2"],     3),
            "MAPE (%)":   round(results["lr_metrics"]["mape"],   2),
            "CI ±1σ ($)": round(results["lr_metrics"]["resid_std"], 2),
        },
        {
            "Model":      "🟠 XGBoost",
            "RMSE ($)":   round(results["xgb_metrics"]["rmse"],  2),
            "MAE ($)":    round(results["xgb_metrics"]["mae"],   2),
            "R²":         round(results["xgb_metrics"]["r2"],    3),
            "MAPE (%)":   round(results["xgb_metrics"]["mape"],  2),
            "CI ±1σ ($)": round(results["xgb_metrics"]["resid_std"], 2),
        },
        {
            "Model":      "🟣 LSTM",
            "RMSE ($)":   round(results["lstm_metrics"]["rmse"], 2),
            "MAE ($)":    round(results["lstm_metrics"]["mae"],  2),
            "R²":         round(results["lstm_metrics"]["r2"],   3),
            "MAPE (%)":   round(results["lstm_metrics"]["mape"], 2),
            "CI ±1σ ($)": round(results["lstm_metrics"]["resid_std"], 2),
        },
    ]
    cmp_df = pd.DataFrame(cmp_rows)

    def highlight_model(row):
        icons = {"Linear Regression": "🔵", "XGBoost": "🟠", "LSTM": "🟣"}
        icon  = icons.get(model_choice, "")
        return ["background-color: #1a3a5c" if icon in row["Model"] else "" for _ in row]

    st.dataframe(
        cmp_df.style.apply(highlight_model, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"🔵 Highlighted = currently selected model ({model_choice}).")

    # Also show RF direction metrics
    d1, d2, d3 = st.columns(3)
    d1.metric("RF Accuracy",    f"{dm['accuracy']*100:.1f}%")
    d2.metric("UP days %",      f"{dm['up_pct']:.1f}%")
    d3.metric("UP Precision",   f"{dm['report']['UP']['precision']*100:.1f}%")

    st.divider()

    # ── Price chart with confidence intervals ─────────────────────────────────
    st.subheader(f"📉 Price History + {model_choice} Predictions + Confidence Interval")

    df_full    = results["full_df"]
    test_dates = results["test_dates"]
    y_actual   = results["y_price_actual"]
    split      = results["train_size"]

    model_pred_map = {
        "Linear Regression": results["y_price_lr"],
        "XGBoost":           results["y_price_xgb"],
        "LSTM":              results["y_price_lstm"],
    }
    model_color_map = {
        "Linear Regression": "#636EFA",
        "XGBoost":           "#FF7F0E",
        "LSTM":              "#AB63FA",
    }
    y_pred      = model_pred_map[model_choice]
    pred_color  = model_color_map[model_choice]
    resid_std   = results[{
        "Linear Regression": "lr_metrics",
        "XGBoost":           "xgb_metrics",
        "LSTM":              "lstm_metrics",
    }[model_choice]]["resid_std"]

    ci_upper = y_pred + resid_std
    ci_lower = y_pred - resid_std

    fig_price = make_subplots(
        rows=2, cols=1,
        row_heights=[0.78, 0.22],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["Close Price + Predictions + CI", "Volume"],
    )

    # Train data
    fig_price.add_trace(go.Scatter(
        x=df_full.index[:split], y=df_full["Close"].values[:split],
        name="Train", line=dict(color="#888888", width=1.2), opacity=0.6,
    ), row=1, col=1)

    # Confidence interval shaded band
    fig_price.add_trace(go.Scatter(
        x=list(test_dates) + list(test_dates)[::-1],
        y=list(ci_upper) + list(ci_lower)[::-1],
        fill="toself",
        fillcolor=f"rgba({','.join(str(int(pred_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1σ CI",
        showlegend=True,
    ), row=1, col=1)

    # Actual test prices
    fig_price.add_trace(go.Scatter(
        x=test_dates, y=y_actual,
        name="Actual", line=dict(color="#00CC96", width=2),
    ), row=1, col=1)

    # Selected model predictions
    fig_price.add_trace(go.Scatter(
        x=test_dates, y=y_pred,
        name=f"{model_choice}", line=dict(color=pred_color, width=2, dash="dash"),
    ), row=1, col=1)

    # MA lines
    fig_price.add_trace(go.Scatter(
        x=df_full.index, y=df_full["MA20"],
        name="MA20", line=dict(color="#FFA15A", width=1, dash="dot"), opacity=0.7,
    ), row=1, col=1)

    # Volume
    fig_price.add_trace(go.Bar(
        x=df_full.index, y=df_full["Volume"],
        name="Volume", marker_color="#636EFA", opacity=0.35,
    ), row=2, col=1)

    fig_price.update_layout(
        height=540, hovermode="x unified",
        legend=dict(orientation="h", y=1.06),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig_price.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig_price.update_yaxes(title_text="Volume",    row=2, col=1)
    st.plotly_chart(fig_price, use_container_width=True)

    # ── Actual vs predicted scatter ───────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("🎯 Actual vs Predicted")
        min_v = min(y_actual.min(), y_pred.min())
        max_v = max(y_actual.max(), y_pred.max())
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v],
            mode="lines", name="Perfect",
            line=dict(color="grey", dash="dash"),
        ))
        fig_sc.add_trace(go.Scatter(
            x=y_actual, y=y_pred, mode="markers",
            name=model_choice,
            marker=dict(color=pred_color, size=5, opacity=0.6),
        ))
        fig_sc.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="Actual ($)", yaxis_title="Predicted ($)")
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_r:
        st.subheader("📊 Direction Confidence")
        y_dir_actual = results["y_dir_actual"]
        y_dir_prob   = results["y_dir_prob"]
        correct = (results["y_dir_pred"] == y_dir_actual)
        colors  = ["#00CC96" if c else "#EF553B" for c in correct]
        fig_dir = go.Figure()
        fig_dir.add_trace(go.Scatter(
            x=list(range(len(y_dir_prob))), y=y_dir_prob,
            mode="markers",
            marker=dict(color=colors, size=5, opacity=0.7),
            name="P(UP)",
        ))
        fig_dir.add_hline(y=0.5, line_dash="dash", line_color="grey",
                          annotation_text="50% threshold")
        fig_dir.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="Test Day", yaxis_title="P(UP)",
                              yaxis=dict(range=[0, 1]))
        st.caption("🟢 Correct   🔴 Wrong")
        st.plotly_chart(fig_dir, use_container_width=True)

    st.divider()

    # ── Feature importance ────────────────────────────────────────────────────
    st.subheader("🧠 Feature Importance (Random Forest)")
    importance = results["importance"]
    imp_df = pd.DataFrame({
        "Feature":    list(importance.keys()),
        "Importance": list(importance.values()),
    }).sort_values("Importance", ascending=True)
    fig_imp = go.Figure(go.Bar(
        x=imp_df["Importance"], y=imp_df["Feature"],
        orientation="h",
        marker=dict(color=imp_df["Importance"], colorscale="Viridis"),
    ))
    fig_imp.update_layout(height=300, xaxis_title="Importance",
                          margin=dict(l=0, r=0, t=5, b=0))
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    # ── Candlestick + Bollinger Bands ─────────────────────────────────────────
    st.subheader("🕯️ Candlestick + Bollinger Bands (Last 90 Days)")
    df_r = df_full.tail(90)

    fig_c = go.Figure()
    # BB fill
    fig_c.add_trace(go.Scatter(
        x=list(df_r.index) + list(df_r.index)[::-1],
        y=list(df_r["BB_Upper"]) + list(df_r["BB_Lower"])[::-1],
        fill="toself", fillcolor="rgba(100,100,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="BB Band",
    ))
    fig_c.add_trace(go.Scatter(x=df_r.index, y=df_r["BB_Upper"],
        name="BB Upper", line=dict(color="rgba(100,100,255,0.5)", width=1, dash="dot")))
    fig_c.add_trace(go.Scatter(x=df_r.index, y=df_r["BB_Mid"],
        name="BB Mid", line=dict(color="#FFA15A", width=1.5)))
    fig_c.add_trace(go.Scatter(x=df_r.index, y=df_r["BB_Lower"],
        name="BB Lower", line=dict(color="rgba(100,100,255,0.5)", width=1, dash="dot")))
    fig_c.add_trace(go.Candlestick(
        x=df_r.index,
        open=df_r["Open"], high=df_r["High"],
        low=df_r["Low"],   close=df_r["Close"],
        name="OHLC",
        increasing_line_color="#00CC96",
        decreasing_line_color="#EF553B",
    ))
    fig_c.update_layout(height=420, xaxis_rangeslider_visible=False,
                        hovermode="x unified", legend=dict(orientation="h", y=1.06),
                        margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_c, use_container_width=True)

    # BB Width
    fig_bbw = go.Figure(go.Scatter(
        x=df_r.index, y=df_r["BB_Width"],
        fill="tozeroy", fillcolor="rgba(100,100,255,0.12)",
        line=dict(color="#636EFA", width=1.5), name="BB Width",
    ))
    fig_bbw.update_layout(height=150, margin=dict(l=0, r=0, t=5, b=0),
                          yaxis_title="Width")
    st.plotly_chart(fig_bbw, use_container_width=True)
    st.caption("BB squeeze (narrow width) → big move often follows.")

    # ── RSI ───────────────────────────────────────────────────────────────────
    st.subheader("📡 RSI (Last 90 Days)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df_r.index, y=df_r["RSI"],
        name="RSI", line=dict(color="#636EFA", width=2)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(height=240, yaxis=dict(range=[0, 100]),
                          margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_rsi, use_container_width=True)

    # ── MACD ──────────────────────────────────────────────────────────────────
    st.subheader("⚡ MACD (Last 90 Days)")
    fig_macd = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4],
                              shared_xaxes=True, vertical_spacing=0.05,
                              subplot_titles=["MACD vs Signal", "Histogram"])
    fig_macd.add_trace(go.Scatter(x=df_r.index, y=df_r["MACD"],
        name="MACD", line=dict(color="#636EFA", width=2)), row=1, col=1)
    fig_macd.add_trace(go.Scatter(x=df_r.index, y=df_r["MACD_Signal"],
        name="Signal", line=dict(color="#EF553B", width=2)), row=1, col=1)
    fig_macd.add_hline(y=0, line_dash="dot", line_color="grey", row=1, col=1)
    hist_colors = ["#00CC96" if v >= 0 else "#EF553B" for v in df_r["MACD_Hist"]]
    fig_macd.add_trace(go.Bar(x=df_r.index, y=df_r["MACD_Hist"],
        name="Histogram", marker_color=hist_colors, opacity=0.7), row=2, col=1)
    fig_macd.add_hline(y=0, line_dash="dot", line_color="grey", row=2, col=1)
    fig_macd.update_layout(height=380, hovermode="x unified",
                           legend=dict(orientation="h", y=1.08),
                           margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_macd, use_container_width=True)
    st.caption("🟢 Green bars = bullish momentum. 🔴 Red = bearish. Crossovers are key signals.")

    st.divider()

    # ── CSV download ──────────────────────────────────────────────────────────
    st.subheader("💾 Download Predictions")
    export_df = pd.DataFrame({
        "Date":              [d.strftime("%Y-%m-%d") for d in test_dates],
        "Actual_Price":      np.round(y_actual, 4),
        "LR_Predicted":      np.round(results["y_price_lr"],   4),
        "XGB_Predicted":     np.round(results["y_price_xgb"],  4),
        "LSTM_Predicted":    np.round(results["y_price_lstm"], 4),
        "Selected_Model":    [model_choice] * len(y_actual),
        "Selected_Pred":     np.round(y_pred, 4),
        "CI_Upper":          np.round(ci_upper, 4),
        "CI_Lower":          np.round(ci_lower, 4),
        "Actual_Direction":  ["UP" if d == 1 else "DOWN" for d in results["y_dir_actual"]],
        "Pred_Direction":    ["UP" if d == 1 else "DOWN" for d in results["y_dir_pred"]],
        "UP_Probability":    np.round(results["y_dir_prob"], 4),
        "Direction_Correct": results["y_dir_pred"] == results["y_dir_actual"],
    })
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.dataframe(export_df.head(8), use_container_width=True, hide_index=True)
    with col_b:
        st.download_button(
            "⬇️ Download Full CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker_name.replace(' ', '_')}_{period_label.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"{len(export_df)} rows · all 3 model predictions included")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — BACKTEST
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("⚡ Backtesting — Strategy vs Buy & Hold")
    st.caption(
        f"**Strategy:** Buy when {model_choice} direction = UP, sell when DOWN. "
        f"No short selling. No transaction costs. Starting capital: $10,000."
    )

    # Summary metrics
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("Strategy Return",   f"{bt['strat_return']:+.1f}%",
              delta=f"{bt['strat_return'] - bt['bh_return']:+.1f}% vs B&H")
    b2.metric("Buy & Hold Return", f"{bt['bh_return']:+.1f}%")
    b3.metric("Strategy Drawdown", f"{bt['strat_drawdown']:.1f}%")
    b4.metric("B&H Drawdown",      f"{bt['bh_drawdown']:.1f}%")
    b5.metric("Strategy Sharpe",   f"{bt['strat_sharpe']:.2f}")
    b6.metric("# Trades",          bt["n_trades"])

    st.divider()

    # P&L curve
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=list(bt["dates"]), y=bt["strategy_vals"],
        name=f"ML Strategy ({model_choice})",
        line=dict(color="#636EFA", width=2.5),
        fill="tozeroy", fillcolor="rgba(99,110,250,0.07)",
    ))
    fig_bt.add_trace(go.Scatter(
        x=list(bt["dates"]), y=bt["bh_vals"],
        name="Buy & Hold",
        line=dict(color="#FFA15A", width=2.5, dash="dash"),
    ))
    fig_bt.add_hline(y=bt["initial_capital"], line_dash="dot",
                     line_color="grey", annotation_text="Starting $10,000")
    fig_bt.update_layout(
        height=440,
        hovermode="x unified",
        yaxis_title="Portfolio Value ($)",
        legend=dict(orientation="h", y=1.06),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # Relative performance
    relative = bt["strategy_vals"] / bt["bh_vals"]
    fig_rel  = go.Figure(go.Scatter(
        x=list(bt["dates"]), y=relative,
        fill="tozeroy",
        fillcolor=f"rgba({'99,200,100' if relative[-1] >= 1 else '250,99,99'},0.12)",
        line=dict(
            color="#00CC96" if relative[-1] >= 1 else "#EF553B",
            width=1.5,
        ),
        name="Strategy / B&H ratio",
    ))
    fig_rel.add_hline(y=1.0, line_dash="dash", line_color="grey",
                      annotation_text="Equal performance")
    fig_rel.update_layout(height=200, yaxis_title="Ratio",
                          margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_rel, use_container_width=True)
    st.caption(
        "**Ratio > 1** = strategy outperforming buy & hold. "
        "**< 1** = buy & hold winning.  "
        "Note: real trading includes commissions and slippage not modelled here."
    )

    st.divider()
    st.subheader("📊 Period Comparison — How does the model choice affect backtest?")

    @st.cache_data(show_spinner=False)
    def compare_backtest(ticker_name, use_live, n_days):
        rows = []
        r = get_results(ticker_name, use_live, n_days)
        for mc in ["Linear Regression", "XGBoost", "LSTM"]:
            b = run_backtest(r, mc)
            rows.append({
                "Model":            mc,
                "Strategy Return":  f"{b['strat_return']:+.1f}%",
                "B&H Return":       f"{b['bh_return']:+.1f}%",
                "Strategy Drawdown":f"{b['strat_drawdown']:.1f}%",
                "Sharpe":           round(b["strat_sharpe"], 2),
                "# Trades":         b["n_trades"],
            })
        return pd.DataFrame(rows)

    bt_cmp = compare_backtest(ticker_name, use_live, n_days)

    def highlight_bt(row):
        return ["background-color: #1a3a5c" if row["Model"] == model_choice else "" for _ in row]

    st.dataframe(bt_cmp.style.apply(highlight_bt, axis=1),
                 use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARE ASSETS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("🌍 Multi-Asset Normalized Performance")
    st.caption(f"All assets rebased to 100 at start · {period_label}")

    COLORS = {
        "NASDAQ (^IXIC)":  "#636EFA",
        "QQQ ETF":         "#EF553B",
        "Gold (GC=F)":     "#FFD700",
        "Bitcoin (BTC)":   "#FF8C00",
        "S&P 500 (^GSPC)": "#00CC96",
    }

    fig_cmp = go.Figure()
    for col in comp_df.columns:
        fig_cmp.add_trace(go.Scatter(
            x=comp_df.index, y=comp_df[col],
            name=col, line=dict(color=COLORS.get(col, "#aaa"), width=2),
            opacity=1.0 if col == ticker_name else 0.45,
        ))
    fig_cmp.add_hline(y=100, line_dash="dot", line_color="grey",
                      annotation_text="Start (100)")
    fig_cmp.update_layout(
        height=420, hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
        yaxis_title="Normalized Price (start=100)",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    if not comp_df.empty:
        final_ret = (comp_df.iloc[-1] - 100).sort_values(ascending=False)
        best  = final_ret.index[0]
        worst = final_ret.index[-1]
        st.caption(
            f"📈 Best: **{best}** ({final_ret[best]:+.1f}%)   "
            f"📉 Worst: **{worst}** ({final_ret[worst]:+.1f}%)"
        )

    # Returns table
    st.subheader("📋 Performance Summary Table")
    ret_rows = []
    for col in comp_df.columns:
        series = comp_df[col].dropna()
        if len(series) < 2:
            continue
        total_ret = float(series.iloc[-1] - 100)
        daily_ret = series.pct_change().dropna()
        vol       = float(daily_ret.std() * np.sqrt(252) * 100)
        sharpe    = (daily_ret.mean() * 252 - 0.04) / (daily_ret.std() * np.sqrt(252) + 1e-9)
        ret_rows.append({
            "Asset":          col,
            "Total Return":   f"{total_ret:+.1f}%",
            "Ann. Volatility":f"{vol:.1f}%",
            "Sharpe Ratio":   round(float(sharpe), 2),
            "Current (norm)": round(float(series.iloc[-1]), 1),
        })
    st.dataframe(pd.DataFrame(ret_rows), use_container_width=True, hide_index=True)

    # Period accuracy comparison
    st.divider()
    st.subheader("📅 How does training period affect model accuracy?")

    @st.cache_data(show_spinner=False)
    def period_accuracy(ticker_name, use_live):
        rows = []
        for label, nd in PERIOD_DAYS.items():
            r = get_results(ticker_name, use_live, nd)
            rows.append({
                "Period":           label,
                "Train days":       r["train_size"],
                "LR RMSE":          round(r["lr_metrics"]["rmse"], 2),
                "XGB RMSE":         round(r["xgb_metrics"]["rmse"], 2),
                "LSTM RMSE":        round(r["lstm_metrics"]["rmse"], 2),
                "RF Accuracy":      f"{r['dir_metrics']['accuracy']*100:.1f}%",
            })
        return pd.DataFrame(rows)

    acc_df = period_accuracy(ticker_name, use_live)

    def hl_period(row):
        return ["background-color: #1a3a5c" if row["Period"] == period_label else "" for _ in row]

    st.dataframe(acc_df.style.apply(hl_period, axis=1),
                 use_container_width=True, hide_index=True)
    st.caption("Highlighted = currently selected period.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PORTFOLIO OPTIMISER
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("💼 Portfolio Optimiser — Efficient Frontier")
    st.caption(
        "Select 2–5 assets. We run 3,000 random portfolios + scipy optimisation "
        "to find the allocation that maximises the Sharpe ratio."
    )

    all_assets = list(TICKERS.keys())
    selected   = st.multiselect(
        "Select assets to include in portfolio",
        options=all_assets,
        default=["S&P 500 (^GSPC)", "Gold (GC=F)", "Bitcoin (BTC)"],
    )

    if len(selected) < 2:
        st.warning("Select at least 2 assets to run the optimiser.")
    elif len(selected) > 5:
        st.warning("Maximum 5 assets — remove some to continue.")
    else:
        with st.spinner("Running Monte Carlo + optimisation…"):
            opt = get_portfolio(tuple(selected), use_live, n_days)

        # Optimal allocation donut chart
        col_p1, col_p2 = st.columns([1, 1])

        with col_p1:
            st.subheader("🏆 Optimal Allocation (Max Sharpe)")
            opt_w = opt["opt_weights"]
            fig_pie = go.Figure(go.Pie(
                labels=list(opt_w.keys()),
                values=[v * 100 for v in opt_w.values()],
                hole=0.45,
                marker=dict(colors=["#636EFA","#EF553B","#FFD700","#00CC96","#AB63FA"]),
                textinfo="label+percent",
            ))
            fig_pie.update_layout(height=340, margin=dict(l=0, r=0, t=20, b=0),
                                  showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

            # Key stats
            p1, p2, p3 = st.columns(3)
            p1.metric("Expected Return", f"{opt['opt_return']*100:.1f}%/yr")
            p2.metric("Volatility",      f"{opt['opt_vol']*100:.1f}%/yr")
            p3.metric("Sharpe Ratio",    f"{opt['opt_sharpe']:.2f}")

            # Allocation table
            alloc_df = pd.DataFrame([
                {"Asset": k, "Weight": f"{v*100:.1f}%", "Raw": v}
                for k, v in opt_w.items()
            ]).sort_values("Raw", ascending=False).drop("Raw", axis=1)
            st.dataframe(alloc_df, use_container_width=True, hide_index=True)

        with col_p2:
            st.subheader("📈 Efficient Frontier (Monte Carlo)")
            sim_returns = opt["sim_returns"]
            sim_vols    = opt["sim_vols"]
            sim_sharpes = opt["sim_sharpes"]

            fig_ef = go.Figure()
            # Simulated portfolios coloured by Sharpe
            fig_ef.add_trace(go.Scatter(
                x=sim_vols * 100, y=sim_returns * 100,
                mode="markers",
                marker=dict(
                    color=sim_sharpes,
                    colorscale="Viridis",
                    size=4, opacity=0.5,
                    colorbar=dict(title="Sharpe", thickness=12),
                ),
                name="Simulated",
                hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%",
            ))
            # Optimal portfolio star
            fig_ef.add_trace(go.Scatter(
                x=[opt["opt_vol"] * 100],
                y=[opt["opt_return"] * 100],
                mode="markers",
                marker=dict(symbol="star", size=18, color="#FFD700",
                            line=dict(color="#000", width=1)),
                name="⭐ Max Sharpe",
            ))
            # Equal-weight reference
            fig_ef.add_trace(go.Scatter(
                x=[opt["equal_vol"] * 100],
                y=[opt["equal_return"] * 100],
                mode="markers",
                marker=dict(symbol="diamond", size=12, color="#EF553B"),
                name="Equal Weight",
            ))
            fig_ef.update_layout(
                height=340,
                xaxis_title="Annual Volatility (%)",
                yaxis_title="Annual Return (%)",
                legend=dict(orientation="h", y=1.06),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_ef, use_container_width=True)

        st.caption(
            "⭐ Gold star = max Sharpe portfolio. 🔴 Diamond = equal-weight baseline. "
            "Dots coloured by Sharpe ratio (yellow = best). "
            "Note: based on synthetic/historical data — past performance ≠ future results."
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — SENTIMENT
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.title("📰 News Sentiment Analysis")
    st.caption(f"Asset: **{ticker_name}** · Powered by TextBlob NLP")

    if not news_api_key:
        st.info(
            "💡 Add a free NewsAPI key in the sidebar to fetch real headlines. "
            "Get one at **newsapi.org** (free tier: 100 requests/day). "
            "Currently showing sample headlines."
        )

    # Overall sentiment meter
    score     = sentiment["avg_score"]
    overall   = sentiment["overall"]
    meter_pct = int((score + 1) / 2 * 100)  # map -1..+1 → 0..100

    sent_col = "#00CC96" if score > 0.05 else ("#EF553B" if score < -0.05 else "#888888")
    st.subheader(f"Overall Sentiment: {overall}")

    # Progress bar as meter
    col_m1, col_m2, col_m3 = st.columns([1, 3, 1])
    with col_m1:
        st.markdown("🔴 **Bearish**")
    with col_m2:
        st.progress(meter_pct / 100)
    with col_m3:
        st.markdown("🟢 **Bullish**")

    s1, s2, s3 = st.columns(3)
    s1.metric("Sentiment Score", f"{score:+.3f}", help="-1 = very negative, +1 = very positive")
    s2.metric("Headlines Analysed", len(sentiment["headlines"]))
    s3.metric("Data Source", "Live (NewsAPI)" if sentiment["is_live"] else "Sample headlines")

    st.divider()

    # Polarity distribution chart
    polarities = [h["polarity"] for h in sentiment["headlines"]]
    fig_sent = go.Figure()
    bar_colors = ["#00CC96" if p > 0.05 else "#EF553B" if p < -0.05 else "#888888"
                  for p in polarities]
    fig_sent.add_trace(go.Bar(
        x=[f"#{i+1}" for i in range(len(polarities))],
        y=polarities,
        marker_color=bar_colors,
        name="Polarity",
    ))
    fig_sent.add_hline(y=0, line_dash="dot", line_color="grey")
    fig_sent.add_hline(y=score, line_dash="dash", line_color="#FFA15A",
                       annotation_text=f"Avg: {score:+.3f}")
    fig_sent.update_layout(
        height=260,
        yaxis_title="Polarity Score",
        yaxis=dict(range=[-1, 1]),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_sent, use_container_width=True)

    # Headline table
    st.subheader("📋 Headline-by-Headline Breakdown")
    headlines_df = pd.DataFrame(sentiment["headlines"])
    headlines_df = headlines_df[["label", "polarity", "subjectivity", "headline"]]
    headlines_df.columns = ["Sentiment", "Polarity", "Subjectivity", "Headline"]

    def color_sent(val):
        if "Bullish" in str(val):
            return "color: #00CC96"
        elif "Bearish" in str(val):
            return "color: #EF553B"
        return "color: #888888"

    st.dataframe(
        headlines_df.style.applymap(color_sent, subset=["Sentiment"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "**Polarity**: -1 (very negative) to +1 (very positive) · "
        "**Subjectivity**: 0 (factual) to 1 (opinion-based)"
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Python · scikit-learn · XGBoost · NumPy LSTM · Streamlit · Plotly · TextBlob · scipy  "
    "| ⚠️ Educational purposes only — not financial advice."
)
