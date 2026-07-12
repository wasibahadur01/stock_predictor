"""
alerts.py — Daily alert system for the stock price predictor.

Runs all 5 assets through the model and sends a Telegram message
when any asset has predicted move confidence above 70%.

Setup:
  1. Create a Telegram bot via @BotFather → get BOT_TOKEN
  2. Send any message to your bot → get CHAT_ID from:
       https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
  3. Set environment variables:
       export TELEGRAM_BOT_TOKEN="your_token_here"
       export TELEGRAM_CHAT_ID="your_chat_id_here"

Run manually:
  python alerts.py

Schedule daily (Linux cron, runs at 8am):
  0 8 * * 1-5 /path/to/venv/bin/python /path/to/alerts.py >> /tmp/alerts.log 2>&1
"""

import os
import urllib.request
import urllib.parse
import json
from datetime import datetime

from model import TICKERS, load_data, train_models, predict_next_day

# ── Config ────────────────────────────────────────────────────────────────────
BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID",   "YOUR_CHAT_ID_HERE")
CONFIDENCE_THRESHOLD = 0.70   # Only alert when confidence > 70%
MODEL_CHOICE = "Linear Regression"
USE_LIVE     = False           # Set True when you have internet + yfinance


# ── Telegram sender ───────────────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    if BOT_TOKEN in ("YOUR_BOT_TOKEN_HERE", "") or CHAT_ID in ("YOUR_CHAT_ID_HERE", ""):
        print("[DRY RUN] Telegram not configured — printing message instead:")
        print(message)
        return False

    try:
        url  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id":    CHAT_ID,
            "text":       message,
            "parse_mode": "Markdown",
        }).encode("utf-8")
        req  = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")
        return False


# ── Alert runner ──────────────────────────────────────────────────────────────

def run_alerts(n_days: int = 252):
    now     = datetime.now().strftime("%Y-%m-%d %H:%M")
    alerts  = []
    summary_lines = [f"📈 *Stock Predictor Daily Report*\n🕐 {now}\n"]

    print(f"[{now}] Running alerts for {len(TICKERS)} assets...")

    for ticker_name in TICKERS:
        try:
            df, is_live  = load_data(ticker_name, use_live=USE_LIVE, n_days=n_days)
            results      = train_models(df)
            pred         = predict_next_day(results, MODEL_CHOICE)

            conf       = pred["confidence"]
            direction  = pred["direction"]
            cur_price  = pred["current_price"]
            pred_price = pred["predicted_price"]
            change_pct = pred["change_pct"]

            direction_emoji = "🟢" if "UP" in direction else "🔴"
            data_src        = "live" if is_live else "synthetic"

            summary_lines.append(
                f"{direction_emoji} *{ticker_name}*\n"
                f"   Current: ${cur_price:,.2f} → Pred: ${pred_price:,.2f} "
                f"({change_pct:+.2f}%)\n"
                f"   Direction: {direction} | Confidence: {conf*100:.1f}% | Source: {data_src}\n"
            )

            print(f"  {ticker_name}: {direction} | conf={conf*100:.1f}% | Δ{change_pct:+.2f}%")

            # Only trigger alert if confidence exceeds threshold
            if conf >= CONFIDENCE_THRESHOLD:
                alert_msg = (
                    f"🚨 *High-Confidence Alert!*\n"
                    f"Asset: *{ticker_name}*\n"
                    f"Direction: *{direction}*\n"
                    f"Confidence: *{conf*100:.1f}%* (threshold: {CONFIDENCE_THRESHOLD*100:.0f}%)\n"
                    f"Current price: ${cur_price:,.2f}\n"
                    f"Predicted:     ${pred_price:,.2f} ({change_pct:+.2f}%)\n"
                    f"Model: {MODEL_CHOICE} | Data: {data_src}\n"
                    f"⚠️ _For educational purposes only — not financial advice._"
                )
                alerts.append(alert_msg)

        except Exception as e:
            print(f"  [ERROR] {ticker_name}: {e}")
            summary_lines.append(f"⚠️ *{ticker_name}*: error — {e}\n")

    # Send individual high-confidence alerts
    if alerts:
        print(f"\n[ALERTS] {len(alerts)} high-confidence signal(s) found — sending Telegram...")
        for msg in alerts:
            success = send_telegram(msg)
            print(f"  Sent: {success}")
    else:
        print(f"\n[ALERTS] No assets exceeded {CONFIDENCE_THRESHOLD*100:.0f}% confidence threshold.")

    # Send daily summary regardless
    summary_lines.append(
        f"\n_{len(alerts)} high-confidence alert(s) sent today._\n"
        f"_Threshold: {CONFIDENCE_THRESHOLD*100:.0f}% | Model: {MODEL_CHOICE}_\n"
        f"⚠️ _Educational only — not financial advice._"
    )
    summary_msg = "\n".join(summary_lines)
    print("\n--- Daily Summary ---")
    print(summary_msg)
    send_telegram(summary_msg)
    return alerts


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    triggered = run_alerts()
    print(f"\nDone. {len(triggered)} alert(s) triggered.")
