"""Simple signal backtesting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_signal_backtest(
    predictions: pd.DataFrame,
    initial_capital: float = 10_000.0,
    transaction_cost_pct: float = 0.001,
) -> pd.DataFrame:
    """Backtest a simple one-day strategy from model direction predictions.

    Strategy rule:
    - If the model predicts UP, stay invested for the next day.
    - If the model predicts DOWN, stay in cash for the next day.

    This is intentionally simple for learning. It ignores slippage, taxes,
    borrow costs, spread, and intraday execution quality.
    """

    required = ["Current_Close", "Actual_Close", "Predicted_Direction"]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"Predictions table missing columns: {', '.join(missing)}")

    df = predictions[required].dropna().copy()
    if df.empty:
        raise ValueError("No predictions available for backtesting.")

    next_day_return = (df["Actual_Close"] / df["Current_Close"]) - 1
    signal = df["Predicted_Direction"].astype(int)

    # Transaction cost is charged when the signal changes between cash and invested.
    signal_change = signal.diff().abs().fillna(signal.iloc[0]).clip(upper=1)
    strategy_return = (signal * next_day_return) - (signal_change * transaction_cost_pct)
    buy_hold_return = next_day_return

    result = pd.DataFrame(index=df.index)
    result["Signal"] = signal
    result["Daily_Strategy_Return"] = strategy_return
    result["Daily_Buy_Hold_Return"] = buy_hold_return
    result["Strategy_Equity"] = initial_capital * (1 + strategy_return).cumprod()
    result["Buy_Hold_Equity"] = initial_capital * (1 + buy_hold_return).cumprod()
    result["Strategy_PnL"] = result["Strategy_Equity"] - initial_capital
    result["Buy_Hold_PnL"] = result["Buy_Hold_Equity"] - initial_capital

    return result.replace([np.inf, -np.inf], np.nan).dropna()
