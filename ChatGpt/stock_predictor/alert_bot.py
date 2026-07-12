"""Daily Telegram alert script for stock_predictor.

Run manually:
    python alert_bot.py

Or schedule it with cron, Windows Task Scheduler, GitHub Actions, or any server.
Required environment variables:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
Optional environment variables:
    ALERT_CONFIDENCE_THRESHOLD=0.70
    ALERT_LOOKBACK_DAYS=730
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import requests

from data import fetch_raw_price_data
from features import create_model_dataset, get_latest_feature_row
from models import predict_tomorrow, train_and_evaluate_all

ASSETS = {
    "^IXIC": "NASDAQ Composite",
    "QQQ": "QQQ Nasdaq 100 ETF",
    "SPY": "SPY S&P 500 ETF",
    "GLD": "GLD Gold ETF",
    "BTC-USD": "Bitcoin",
}


def format_price(value: float, symbol: str) -> str:
    decimals = 0 if symbol == "BTC-USD" else 2
    return f"${value:,.{decimals}f}"


def send_telegram_message(token: str, chat_id: str, message: str) -> None:
    """Send a Telegram message using Bot API sendMessage."""

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(
        url,
        data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
        timeout=20,
    )
    response.raise_for_status()


def build_alerts(threshold: float, lookback_days: int) -> list[str]:
    """Train models for all assets and return alert message lines."""

    alerts: list[str] = []
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    for symbol, label in ASSETS.items():
        try:
            price_data = fetch_raw_price_data(symbol, start_date, end_date)
            model_data = create_model_dataset(price_data)
            bundle = train_and_evaluate_all(model_data, include_lstm=False)
            latest_features = get_latest_feature_row(price_data)

            # Random Forest gives a probability-based direction confidence.
            selected_model = "Random Forest" if "Random Forest" in bundle.results else "Linear Regression"
            latest_close = float(price_data["Close"].iloc[-1])
            prediction = predict_tomorrow(
                bundle,
                latest_features,
                model_data,
                selected_model,
                latest_close=latest_close,
            )

            confidence = float(prediction["confidence"])
            if confidence < threshold:
                continue

            predicted_price = float(prediction["predicted_price"])
            predicted_direction = int(prediction["predicted_direction"])
            move_pct = ((predicted_price / latest_close) - 1) * 100
            direction_label = "UP 📈" if predicted_direction == 1 else "DOWN 📉"

            alerts.append(
                f"*{label}* (`{symbol}`)\n"
                f"Model: {selected_model}\n"
                f"Prediction: *{direction_label}*\n"
                f"Confidence: *{confidence:.1%}*\n"
                f"Latest close: {format_price(latest_close, symbol)}\n"
                f"Predicted close: {format_price(predicted_price, symbol)}\n"
                f"Predicted move: {move_pct:+.2f}%"
            )
        except Exception as exc:
            alerts.append(f"*{label}* (`{symbol}`): skipped because {exc}")

    return alerts


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    threshold = float(os.getenv("ALERT_CONFIDENCE_THRESHOLD", "0.70"))
    lookback_days = int(os.getenv("ALERT_LOOKBACK_DAYS", "730"))

    if not token or not chat_id:
        raise SystemExit(
            "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variable."
        )

    alerts = build_alerts(threshold=threshold, lookback_days=lookback_days)
    if not alerts:
        message = f"Stock Predictor: no assets crossed {threshold:.0%} confidence today."
    else:
        message = "*Stock Predictor Daily Alerts*\n\n" + "\n\n---\n\n".join(alerts)

    send_telegram_message(token, chat_id, message)
    print("Alert script finished.")


if __name__ == "__main__":
    main()
