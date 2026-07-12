"""Feature engineering and technical indicators for stock prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "SMA_20",
    "SMA_50",
    "RSI_14",
    "Momentum_10",
    "Volume_Change_Pct",
    "Daily_Return_Pct",
    "Distance_SMA20_Pct",
]

LSTM_FEATURE_COLUMNS = FEATURE_COLUMNS + ["Close"]
TARGET_CLOSE = "Target_Close_Next_Day"
TARGET_DIRECTION = "Target_Direction"
TARGET_DATE = "Target_Date"


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index using simple rolling averages."""

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # If average loss is zero and average gain is positive, RSI is conventionally 100.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100)
    return rsi


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators used by the dashboard and models.

    The function is deliberately defensive: Yahoo Finance can return zero or
    missing volume for some index symbols. In that case the volume-change
    feature is filled with 0 so one weak data column does not destroy the whole
    training set.
    """

    df = data.copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df.get("Volume", 0), errors="coerce").fillna(0)

    # Required model features.
    df["SMA_20"] = df["Close"].rolling(window=20, min_periods=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=50).mean()
    df["RSI_14"] = calculate_rsi(df["Close"], window=14)
    df["Momentum_10"] = df["Close"].pct_change(periods=10, fill_method=None) * 100

    volume_change = df["Volume"].replace(0, np.nan).pct_change(fill_method=None) * 100
    df["Volume_Change_Pct"] = volume_change.replace([np.inf, -np.inf], np.nan).fillna(0)

    df["Daily_Return_Pct"] = df["Close"].pct_change(fill_method=None) * 100
    df["Distance_SMA20_Pct"] = ((df["Close"] - df["SMA_20"]) / df["SMA_20"]) * 100

    # Bollinger Bands: 20-day SMA +/- 2 rolling standard deviations.
    rolling_std_20 = df["Close"].rolling(window=20, min_periods=20).std()
    df["BB_Middle"] = df["SMA_20"]
    df["BB_Upper"] = df["BB_Middle"] + (2 * rolling_std_20)
    df["BB_Lower"] = df["BB_Middle"] - (2 * rolling_std_20)

    # MACD: 12/26 EMA line, 9 EMA signal, and histogram.
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_Line"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD_Line"] - df["MACD_Signal"]

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def create_model_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Create a cleaned model dataset with features and next-day targets."""

    df = add_technical_indicators(data)

    df[TARGET_CLOSE] = df["Close"].shift(-1)
    df[TARGET_DIRECTION] = (df[TARGET_CLOSE] > df["Close"]).astype(int)
    df[TARGET_DATE] = df.index.to_series().shift(-1)

    columns_needed = list(
        dict.fromkeys(
            FEATURE_COLUMNS + LSTM_FEATURE_COLUMNS + [TARGET_CLOSE, TARGET_DIRECTION, TARGET_DATE]
        )
    )
    df = df.dropna(subset=columns_needed).copy()
    df[TARGET_DATE] = pd.to_datetime(df[TARGET_DATE])

    return df


def get_latest_feature_row(data: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent complete feature row for tomorrow's prediction."""

    df = add_technical_indicators(data)
    latest = df.dropna(subset=FEATURE_COLUMNS).tail(1)

    if latest.empty:
        raise ValueError(
            "Not enough price history to calculate the latest technical indicators. "
            "Choose a wider date range."
        )

    return latest[FEATURE_COLUMNS]


def get_latest_lstm_sequence(data: pd.DataFrame, sequence_length: int = 60) -> pd.DataFrame:
    """Return the latest sequence of LSTM features, including the latest market day."""

    df = add_technical_indicators(data).dropna(subset=LSTM_FEATURE_COLUMNS).copy()
    if len(df) < sequence_length:
        raise ValueError(
            f"Need at least {sequence_length} clean rows to build the LSTM sequence."
        )
    return df[LSTM_FEATURE_COLUMNS].tail(sequence_length)
