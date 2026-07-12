"""NewsAPI + TextBlob sentiment helper functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
import requests
from textblob import TextBlob

NEWSAPI_URL = "https://newsapi.org/v2/everything"

NEWS_QUERIES: Dict[str, str] = {
    "^IXIC": '"Nasdaq Composite" OR Nasdaq',
    "QQQ": 'QQQ OR "Nasdaq 100"',
    "SPY": 'SPY OR "S&P 500"',
    "GLD": 'GLD OR "gold ETF" OR "gold price"',
    "BTC-USD": 'Bitcoin OR BTC OR cryptocurrency',
}


@dataclass
class SentimentResult:
    """Container for scored headlines."""

    headlines: pd.DataFrame
    average_score: float
    label: str


def fetch_news_sentiment(symbol: str, api_key: str, page_size: int = 10) -> SentimentResult:
    """Fetch the latest headlines for an asset and score them with TextBlob."""

    if not api_key:
        raise ValueError("Add NEWSAPI_KEY to Streamlit secrets or your environment.")

    query = NEWS_QUERIES.get(symbol, symbol)
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }

    response = requests.get(NEWSAPI_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()

    articles = payload.get("articles", [])[:page_size]
    rows = []
    for article in articles:
        title = article.get("title") or ""
        description = article.get("description") or ""
        text = f"{title}. {description}".strip()
        polarity = float(TextBlob(text).sentiment.polarity) if text else 0.0
        rows.append(
            {
                "Published": article.get("publishedAt"),
                "Source": (article.get("source") or {}).get("name"),
                "Headline": title,
                "Sentiment": polarity,
                "URL": article.get("url"),
            }
        )

    headlines = pd.DataFrame(rows)
    average_score = float(headlines["Sentiment"].mean()) if not headlines.empty else 0.0
    label = sentiment_label(average_score)
    return SentimentResult(headlines=headlines, average_score=average_score, label=label)


def sentiment_label(score: float) -> str:
    """Convert polarity score into a dashboard-friendly label."""

    if score > 0.05:
        return "Bullish"
    if score < -0.05:
        return "Bearish"
    return "Neutral"
