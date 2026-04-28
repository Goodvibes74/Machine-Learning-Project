"""
NLP Sentiment Analysis Module

Uses FinBERT (ProsusAI/finbert) to score financial news headlines and
produce daily sentiment features for the stock prediction pipeline.

Two news sources are supported:
  - Finnhub  (https://finnhub.io)  — recommended, free tier available
  - NewsAPI  (https://newsapi.org) — alternative

Set SENTIMENT_API_KEY in config.py and the pipeline will automatically
replace the price-based proxy with real NLP sentiment scores.
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Optional-import guards — the rest of the pipeline runs fine without these
# ---------------------------------------------------------------------------
try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    from transformers import pipeline as _hf_pipeline
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False


_FINNHUB_BASE = "https://finnhub.io/api/v1"
_NEWSAPI_BASE  = "https://newsapi.org/v2"

_finbert_pipe = None  # Lazy singleton — loaded once on first call to score_headlines


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_finbert():
    global _finbert_pipe
    if _finbert_pipe is None:
        if not _TRANSFORMERS_OK:
            raise ImportError(
                "The 'transformers' library is required for NLP sentiment analysis.\n"
                "Install it with:  pip install transformers torch"
            )
        print("  [sentiment] Loading FinBERT model (first run downloads ~400 MB)…")
        _finbert_pipe = _hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            return_all_scores=True,
        )
        print("  [sentiment] FinBERT ready.")
    return _finbert_pipe


def _fetch_from_finnhub(ticker, start_date, end_date, api_key):
    url = f"{_FINNHUB_BASE}/company-news"
    params = {
        "symbol": ticker,
        "from":   start_date,
        "to":     end_date,
        "token":  api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json()
        headlines = []
        for art in articles:
            ts = art.get("datetime", 0)
            date_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d') if ts else None
            headline  = (art.get("headline") or "").strip()
            if date_str and headline:
                headlines.append({"date": date_str, "headline": headline})
        return headlines
    except Exception as exc:
        print(f"  [sentiment] Finnhub error for {ticker}: {exc}")
        return []


def _fetch_from_newsapi(ticker, start_date, end_date, api_key):
    url = f"{_NEWSAPI_BASE}/everything"
    params = {
        "q":        ticker,
        "from":     start_date,
        "to":       end_date,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": 100,
        "apiKey":   api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        headlines = []
        for art in articles:
            pub = art.get("publishedAt", "")
            date_str = pub[:10] if pub else None
            headline  = (art.get("title") or "").strip()
            if date_str and headline:
                headlines.append({"date": date_str, "headline": headline})
        return headlines
    except Exception as exc:
        print(f"  [sentiment] NewsAPI error for {ticker}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_news_headlines(ticker, start_date, end_date, api_key=None, source="finnhub"):
    """
    Fetch news headlines for a single ticker over a date range.

    Args:
        ticker     (str):            Stock symbol, e.g. 'AAPL'
        start_date (str|datetime):   Start of date range (YYYY-MM-DD)
        end_date   (str|datetime):   End of date range   (YYYY-MM-DD)
        api_key    (str|None):       Finnhub or NewsAPI key.
                                     Returns [] immediately if None.
        source     (str):            'finnhub' (default) or 'newsapi'

    Returns:
        list[dict]:  [{'date': 'YYYY-MM-DD', 'headline': '...'}, …]
                     Empty list when no key is provided or on fetch error.
    """
    if not api_key:
        print(f"  [sentiment] No API key — skipping news fetch for {ticker}.")
        return []

    if not _REQUESTS_OK:
        print("  [sentiment] 'requests' library missing — cannot fetch news.")
        return []

    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')

    if source == "finnhub":
        return _fetch_from_finnhub(ticker, start_date, end_date, api_key)
    elif source == "newsapi":
        return _fetch_from_newsapi(ticker, start_date, end_date, api_key)
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'finnhub' or 'newsapi'.")


def score_headlines(headlines):
    """
    Run FinBERT on a list of headline dicts and return sentiment scores.

    FinBERT labels each headline as positive / negative / neutral.
    The continuous score returned here is  (P_positive − P_negative),
    ranging from −1 (fully bearish) to +1 (fully bullish).

    Args:
        headlines (list[dict]): Output of fetch_news_headlines().

    Returns:
        pd.DataFrame:  columns ['date', 'headline', 'sentiment_score']
                       Returns an empty DataFrame with correct schema if
                       input is empty or transformers is unavailable.
    """
    _empty = pd.DataFrame(columns=['date', 'headline', 'sentiment_score'])

    if not headlines:
        return _empty

    if not _TRANSFORMERS_OK:
        print("  [sentiment] transformers unavailable — assigning neutral scores.")
        df = pd.DataFrame(headlines)
        df['sentiment_score'] = 0.0
        return df[['date', 'headline', 'sentiment_score']]

    model  = _load_finbert()
    texts  = [h['headline'] for h in headlines]
    scores = []

    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            results = model(batch, truncation=True, max_length=512)
        except Exception as exc:
            print(f"  [sentiment] FinBERT inference error: {exc}")
            results = [[]] * len(batch)

        for result in results:
            if not result:
                scores.append(0.0)
                continue
            by_label = {r['label'].lower(): r['score'] for r in result}
            net = by_label.get('positive', 0.0) - by_label.get('negative', 0.0)
            scores.append(net)

    df = pd.DataFrame(headlines)
    df['sentiment_score'] = scores
    return df[['date', 'headline', 'sentiment_score']]


def aggregate_sentiment(scored_df, all_dates=None):
    """
    Roll up headline-level scores into daily NLP_Sentiment and News_Volume.

    Missing trading days (no news) receive NLP_Sentiment = 0.0 and
    News_Volume = 0, preserving the shape of the time series.

    Args:
        scored_df  (pd.DataFrame):          Output of score_headlines().
        all_dates  (list|pd.DatetimeIndex):  Complete date range to fill.
                                             When provided, every date gets a row.

    Returns:
        pd.DataFrame:  columns ['Date', 'NLP_Sentiment', 'News_Volume']
    """
    if scored_df.empty:
        if all_dates is not None:
            return pd.DataFrame({
                'Date':          pd.to_datetime(all_dates),
                'NLP_Sentiment': 0.0,
                'News_Volume':   0,
            })
        return pd.DataFrame(columns=['Date', 'NLP_Sentiment', 'News_Volume'])

    scored_df         = scored_df.copy()
    scored_df['date'] = pd.to_datetime(scored_df['date'])

    daily = (
        scored_df
        .groupby('date')
        .agg(
            NLP_Sentiment=('sentiment_score', 'mean'),
            News_Volume  =('sentiment_score', 'count'),
        )
        .reset_index()
        .rename(columns={'date': 'Date'})
    )

    if all_dates is not None:
        dates_df = pd.DataFrame({'Date': pd.to_datetime(all_dates)})
        daily    = dates_df.merge(daily, on='Date', how='left')
        daily['NLP_Sentiment'] = daily['NLP_Sentiment'].fillna(0.0)
        daily['News_Volume']   = daily['News_Volume'].fillna(0).astype(int)

    return daily


def get_sentiment_for_ticker(ticker, start_date, end_date, api_key=None, source="finnhub"):
    """
    Full sentiment pipeline for one ticker: fetch → score → aggregate.

    The returned DataFrame covers every calendar day in [start_date, end_date].
    Days without news are filled with NLP_Sentiment = 0.0 (neutral) and
    News_Volume = 0 so that the downstream merge always has a complete index.

    Args:
        ticker     (str):           Stock symbol
        start_date (str|datetime):  Range start
        end_date   (str|datetime):  Range end
        api_key    (str|None):      Finnhub or NewsAPI key
        source     (str):           'finnhub' (default) or 'newsapi'

    Returns:
        pd.DataFrame:  ['Date', 'NLP_Sentiment', 'News_Volume']
    """
    print(f"  [sentiment] Collecting sentiment for {ticker}…")

    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    headlines      = fetch_news_headlines(ticker, start_date, end_date,
                                          api_key=api_key, source=source)
    print(f"  [sentiment] {len(headlines)} headlines fetched for {ticker}.")

    scored          = score_headlines(headlines)
    daily_sentiment = aggregate_sentiment(scored, all_dates=all_dates)

    return daily_sentiment


# ---------------------------------------------------------------------------
# Standalone test — python src/sentiment_analysis.py
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

    _ticker  = 'AAPL'
    _end     = config.END_DATE
    _start   = (pd.Timestamp(_end) - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    _api_key = config.SENTIMENT_API_KEY
    _source  = config.SENTIMENT_SOURCE

    print(f"\nSentiment module smoke-test  |  {_ticker}  |  {_start} → {_end}")
    print(f"API key configured: {'YES' if _api_key else 'NO  (neutral scores will be returned)'}")
    print("-" * 60)

    _sent = get_sentiment_for_ticker(_ticker, _start, _end,
                                     api_key=_api_key, source=_source)

    print(f"\nSentiment for {_ticker} — last 5 rows:")
    print(_sent.tail())
    print(f"\nMean NLP_Sentiment : {_sent['NLP_Sentiment'].mean():.4f}")
    print(f"Days with news     : {(_sent['News_Volume'] > 0).sum()} / {len(_sent)}")
    print("\nDone.")
