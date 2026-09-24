#!/usr/bin/env python3
"""
News fetching and sentiment scoring for the ML strategy.

- ``AlpacaNewsClient`` pulls headlines/summaries from Alpaca's News API.
- ``FinBertScorer`` scores text with ProsusAI/finbert (optional dependency:
  ``pip install -r requirements-ml.txt``); ``LexiconScorer`` is a
  dependency-free fallback. Both return scores in [-1, +1].
- ``weighted_sentiment`` / ``rolling_sentiment`` aggregate article scores into
  a 24-hour, time-decayed metric using only articles published at or before
  the evaluation time, so the same code is safe for training and live use.
"""

from __future__ import annotations

import bisect
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, List, Optional, Protocol, Sequence

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SENTIMENT_FEATURES = ["sentiment_score", "sentiment_article_count", "sentiment_available"]


@dataclass(frozen=True)
class NewsArticle:
    published_at: datetime
    headline: str
    summary: str = ""
    symbols: tuple = ()

    @property
    def text(self) -> str:
        return f"{self.headline}. {self.summary}".strip(" .")


@dataclass(frozen=True)
class ScoredArticle:
    published_at: datetime
    score: float


@dataclass(frozen=True)
class SentimentSnapshot:
    score: float
    article_count: int
    available: bool

    def as_features(self) -> dict:
        return {
            "sentiment_score": self.score if self.available else float("nan"),
            "sentiment_article_count": float(self.article_count) if self.available else float("nan"),
            "sentiment_available": 1.0 if self.available else 0.0,
        }


UNAVAILABLE = SentimentSnapshot(score=0.0, article_count=0, available=False)


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def _parse_time(value: str) -> datetime:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


class AlpacaNewsClient:
    """Minimal client for GET /v1beta1/news (paginated)."""

    def __init__(self, http_get: Callable = requests.get) -> None:
        self.http_get = http_get
        self.key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY_ID")
        self.secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET")
        self.base = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").rstrip("/")

    @property
    def configured(self) -> bool:
        return bool(self.key and self.secret)

    def fetch(self, symbol: str, start: datetime, end: datetime, max_articles: int = 1000) -> List[NewsArticle]:
        if not self.configured:
            raise ValueError("Alpaca credentials not configured")
        params = {
            "symbols": symbol,
            "start": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "end": end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "limit": 50,
            "sort": "asc",
            "include_content": "false",
        }
        headers = {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}
        articles: List[NewsArticle] = []
        while len(articles) < max_articles:
            resp = self.http_get(f"{self.base}/v1beta1/news", headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            for item in payload.get("news") or []:
                articles.append(
                    NewsArticle(
                        published_at=_parse_time(item["created_at"]),
                        headline=item.get("headline") or "",
                        summary=item.get("summary") or "",
                        symbols=tuple(item.get("symbols") or ()),
                    )
                )
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        return articles[:max_articles]


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

class SentimentScorer(Protocol):
    name: str

    def score(self, texts: Sequence[str]) -> List[float]:
        ...


class LexiconScorer:
    """Dependency-free finance word-list scorer (fallback when FinBERT isn't installed)."""

    name = "lexicon"
    POSITIVE = frozenset(
        """beat beats exceeded exceeds surge surges soar soars rally rallies gain gains jump jumps
        record upgrade upgraded upgrades outperform bullish growth profit profitable strong
        stronger raise raises raised boost boosts approval approved expands expansion win wins
        buyback dividend partnership breakthrough rebound optimistic top tops higher""".split()
    )
    NEGATIVE = frozenset(
        """miss misses missed plunge plunges plummet slump slumps drop drops fall falls fell
        decline declines downgrade downgraded downgrades underperform bearish loss losses weak
        weaker cut cuts lawsuit probe investigation recall fraud bankruptcy default layoffs
        warning warns halt halted delay delayed lower lowers sell-off selloff pessimistic""".split()
    )
    NEGATIONS = frozenset({"not", "no", "never", "without", "fails", "failed"})
    _token = re.compile(r"[a-z][a-z\-']*")

    def score(self, texts: Sequence[str]) -> List[float]:
        return [self._score_one(t) for t in texts]

    def _score_one(self, text: str) -> float:
        tokens = self._token.findall(text.lower())
        total = 0
        for i, tok in enumerate(tokens):
            polarity = (tok in self.POSITIVE) - (tok in self.NEGATIVE)
            if polarity and i > 0 and tokens[i - 1] in self.NEGATIONS:
                polarity = -polarity
            total += polarity
        return math.tanh(total / 2)


class FinBertScorer:
    """ProsusAI/finbert via Hugging Face transformers; score = P(positive) - P(negative)."""

    name = "finbert"
    MODEL_ID = "ProsusAI/finbert"

    def __init__(self, classifier: Optional[Callable] = None, batch_size: int = 16) -> None:
        if classifier is None:
            from transformers import pipeline  # optional heavy dependency

            classifier = pipeline("text-classification", model=self.MODEL_ID, top_k=None, truncation=True)
        self.classifier = classifier
        self.batch_size = batch_size

    def score(self, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        outputs = self.classifier(list(texts), batch_size=self.batch_size)
        scores = []
        for label_scores in outputs:
            probs = {d["label"].lower(): float(d["score"]) for d in label_scores}
            scores.append(max(-1.0, min(1.0, probs.get("positive", 0.0) - probs.get("negative", 0.0))))
        return scores


_SCORER: Optional[SentimentScorer] = None


def get_scorer(preference: Optional[str] = None) -> SentimentScorer:
    """Return a cached scorer: SENTIMENT_MODEL=finbert|lexicon|auto (default auto)."""
    global _SCORER
    preference = (preference or os.getenv("SENTIMENT_MODEL", "auto")).lower()
    if _SCORER is not None and (preference == "auto" or _SCORER.name == preference):
        return _SCORER

    if preference in {"auto", "finbert"}:
        try:
            _SCORER = FinBertScorer()
            return _SCORER
        except Exception as exc:  # ImportError, or model download failure
            level = logging.WARNING if preference == "finbert" else logging.INFO
            logger.log(level, f"FinBERT unavailable ({exc}); using lexicon sentiment scorer")
    _SCORER = LexiconScorer()
    return _SCORER


def score_articles(articles: Iterable[NewsArticle], scorer: SentimentScorer) -> List[ScoredArticle]:
    articles = [a for a in articles if a.text]
    scores = scorer.score([a.text for a in articles])
    return [ScoredArticle(a.published_at, s) for a, s in zip(articles, scores)]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def weighted_sentiment(
    scored: Sequence[ScoredArticle],
    at: datetime,
    *,
    window: timedelta = timedelta(hours=24),
    half_life: timedelta = timedelta(hours=6),
) -> SentimentSnapshot:
    """Time-decayed mean score of articles in (at - window, at]; never looks past ``at``."""
    num = den = 0.0
    count = 0
    for article in scored:
        age = at - article.published_at
        if age < timedelta(0) or age > window:
            continue
        weight = 0.5 ** (age / half_life)
        num += weight * article.score
        den += weight
        count += 1
    if count == 0:
        return SentimentSnapshot(score=0.0, article_count=0, available=True)
    return SentimentSnapshot(score=max(-1.0, min(1.0, num / den)), article_count=count, available=True)


def rolling_sentiment(
    bar_close_times: Sequence[datetime],
    scored: Sequence[ScoredArticle],
    **kwargs,
) -> pd.DataFrame:
    """Sentiment features for each bar, evaluated at that bar's close time."""
    window = kwargs.get("window", timedelta(hours=24))
    ordered = sorted(scored, key=lambda a: a.published_at)
    times = [a.published_at for a in ordered]
    rows = []
    for t in bar_close_times:
        # Only articles in (t - window, t] can contribute; slice them out first.
        lo = bisect.bisect_left(times, t - window)
        hi = bisect.bisect_right(times, t)
        rows.append(weighted_sentiment(ordered[lo:hi], t, **kwargs).as_features())
    return pd.DataFrame(rows, index=pd.DatetimeIndex(bar_close_times), columns=SENTIMENT_FEATURES)


def half_life_from_env() -> timedelta:
    return timedelta(hours=float(os.getenv("SENTIMENT_HALF_LIFE_HOURS", "6")))


def live_sentiment(
    symbol: str,
    *,
    client: Optional[AlpacaNewsClient] = None,
    scorer: Optional[SentimentScorer] = None,
    now: Optional[datetime] = None,
) -> SentimentSnapshot:
    """Current 24h weighted sentiment, or UNAVAILABLE if news can't be fetched."""
    client = client or AlpacaNewsClient()
    now = now or datetime.now(timezone.utc)
    try:
        articles = client.fetch(symbol, now - timedelta(hours=24), now, max_articles=200)
        scored = score_articles(articles, scorer or get_scorer())
    except Exception as exc:
        logger.warning(f"Sentiment unavailable for {symbol}: {exc}")
        return UNAVAILABLE
    return weighted_sentiment(scored, now, half_life=half_life_from_env())
