"""Data loading utilities for the Stock Predictor app.

The Streamlit dashboard uses ``fetch_price_data`` so repeated yfinance calls are
cached. The alert script uses ``fetch_raw_price_data`` because it runs outside a
Streamlit session.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, Optional, Union

import pandas as pd

try:
    import streamlit as st
except Exception:  # Allows non-Streamlit scripts/tests to import this module.
    class _StreamlitFallback:
        @staticmethod
        def cache_data(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

    st = _StreamlitFallback()

try:
    import yfinance as yf
except Exception:  # pragma: no cover - requirements.txt installs this for normal use.
    yf = None

DateLike = Union[str, date, datetime, pd.Timestamp]


def fetch_raw_price_data(
    symbol: str,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    period: str = "3y",
) -> pd.DataFrame:
    """Fetch historical OHLCV price data for one asset from Yahoo Finance.

    Parameters
    ----------
    symbol:
        Yahoo Finance ticker symbol, for example ``QQQ`` or ``BTC-USD``.
    start_date / end_date:
        Optional date range. ``end_date`` is treated as inclusive by this app;
        one day is added before sending it to yfinance because yfinance's end
        date is exclusive.
    period:
        yfinance period string used when no explicit start/end dates are given.
    """

    if yf is None:
        raise ImportError("yfinance is required to download market data. Run: pip install yfinance")

    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("Please choose a valid ticker symbol.")

    download_kwargs = {
        "tickers": symbol,
        "progress": False,
        "auto_adjust": False,
        "threads": False,
    }

    if start_date is not None and end_date is not None:
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date() + timedelta(days=1)
        download_kwargs.update({"start": start, "end": end})
    else:
        download_kwargs.update({"period": period})

    data = yf.download(**download_kwargs)

    if data is None or data.empty:
        raise ValueError(
            f"No data returned for {symbol}. Try a wider date range or another asset."
        )

    data = _flatten_yfinance_columns(data, symbol)
    data = _standardize_ohlcv(data)

    if data.empty:
        raise ValueError(
            f"Downloaded data for {symbol} did not contain usable OHLCV rows."
        )

    return data


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_price_data(
    symbol: str,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    period: str = "3y",
) -> pd.DataFrame:
    """Cached wrapper around ``fetch_raw_price_data`` for the dashboard."""

    return fetch_raw_price_data(symbol, start_date, end_date, period)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_close_prices(
    symbols: tuple[str, ...],
    start_date: DateLike,
    end_date: DateLike,
) -> pd.DataFrame:
    """Fetch adjusted/close prices for several assets and align them by date."""

    close_frames: Dict[str, pd.Series] = {}
    errors: list[str] = []

    for symbol in symbols:
        try:
            data = fetch_raw_price_data(symbol, start_date, end_date)
            close_frames[symbol] = data["Close"].rename(symbol)
        except Exception as exc:  # pragma: no cover - message is shown in UI
            errors.append(f"{symbol}: {exc}")

    if not close_frames:
        raise ValueError("No comparison data could be downloaded. " + "; ".join(errors))

    prices = pd.concat(close_frames.values(), axis=1).sort_index()
    prices = prices.ffill().dropna(how="all")

    if prices.empty:
        raise ValueError("Downloaded comparison data was empty after cleaning.")

    return prices


def _flatten_yfinance_columns(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Handle yfinance single-ticker and multi-index column formats."""

    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        # Newer yfinance versions may return columns like ('Close', 'QQQ').
        for level in range(data.columns.nlevels):
            values = data.columns.get_level_values(level)
            if symbol in values:
                data = data.xs(symbol, axis=1, level=level)
                break
        else:
            # Safe fallback: join column levels into readable names.
            data.columns = [
                "_".join(str(part) for part in col if part) for col in data.columns
            ]

    return data


def _standardize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, data types, and date index."""

    data = data.copy()
    data.columns = [str(col).strip().title() for col in data.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    keep_columns = required + (["Adj Close"] if "Adj Close" in data.columns else [])
    data = data[keep_columns]

    for column in keep_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data.index = pd.to_datetime(data.index)
    try:
        data.index = data.index.tz_localize(None)
    except TypeError:
        # The index is already timezone-naive.
        pass

    data.index.name = "Date"
    data = data.sort_index()
    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    return data
