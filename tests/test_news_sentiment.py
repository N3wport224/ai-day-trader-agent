from __future__ import annotations

import builtins
from datetime import datetime, timedelta, timezone

import pytest

from core import news_sentiment
from core.news_sentiment import (
    AlpacaNewsClient,
    FinBertScorer,
    LexiconScorer,
    NewsArticle,
    ScoredArticle,
    get_scorer,
    live_sentiment,
    rolling_sentiment,
    score_articles,
    weighted_sentiment,
)

NOW = datetime(2026, 3, 2, 18, 0, tzinfo=timezone.utc)


def _at(hours_ago: float, score: float) -> ScoredArticle:
    return ScoredArticle(NOW - timedelta(hours=hours_ago), score)


def test_lexicon_scorer_polarity_and_bounds() -> None:
    scorer = LexiconScorer()
    pos, neg, neutral, negated = scorer.score(
        [
            "Apple beats estimates as profit surges to record",
            "Shares plunge after guidance cut and downgrade",
            "Company schedules annual meeting",
            "Results did not beat expectations",
        ]
    )

    assert 0 < pos <= 1
    assert -1 <= neg < 0
    assert neutral == 0
    assert negated < 0


def test_finbert_scorer_maps_probabilities_to_signed_score() -> None:
    def fake_classifier(texts, batch_size):
        return [
            [{"label": "positive", "score": 0.8}, {"label": "negative", "score": 0.1}, {"label": "neutral", "score": 0.1}],
            [{"label": "Positive", "score": 0.05}, {"label": "Negative", "score": 0.9}, {"label": "Neutral", "score": 0.05}],
        ]

    scores = FinBertScorer(classifier=fake_classifier).score(["good", "bad"])

    assert scores == pytest.approx([0.7, -0.85])


def test_get_scorer_falls_back_to_lexicon_without_transformers(monkeypatch) -> None:
    monkeypatch.setattr(news_sentiment, "_SCORER", None)
    real_import = builtins.__import__

    def no_transformers(name, *args, **kwargs):
        if name.startswith("transformers"):
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_transformers)

    assert get_scorer("auto").name == "lexicon"
    monkeypatch.setattr(news_sentiment, "_SCORER", None)
    assert get_scorer("lexicon").name == "lexicon"


def test_weighted_sentiment_decays_and_ignores_old_or_future_articles() -> None:
    articles = [
        _at(0, 1.0),        # weight 1
        _at(6, -1.0),       # weight 0.5 (one half-life)
        _at(30, -1.0),      # outside 24h window
        _at(-1, -1.0),      # published after `at`: must not leak
    ]

    snap = weighted_sentiment(articles, NOW, half_life=timedelta(hours=6))

    assert snap.article_count == 2
    assert snap.score == pytest.approx((1.0 - 0.5) / 1.5)
    assert snap.available


def test_weighted_sentiment_without_news_is_neutral_but_available() -> None:
    snap = weighted_sentiment([], NOW)

    assert (snap.score, snap.article_count, snap.available) == (0.0, 0, True)


def test_rolling_sentiment_is_causal_per_bar() -> None:
    articles = [_at(5, 0.8), _at(1, -0.6)]
    bar_closes = [NOW - timedelta(hours=h) for h in (6, 3, 0)]

    frame = rolling_sentiment(bar_closes, articles, half_life=timedelta(hours=6))

    assert frame["sentiment_article_count"].tolist() == [0.0, 1.0, 2.0]
    assert frame["sentiment_score"].iloc[1] == pytest.approx(0.8)
    assert frame["sentiment_available"].tolist() == [1.0, 1.0, 1.0]


def test_alpaca_news_client_paginates_and_parses(monkeypatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    pages = [
        {"news": [{"created_at": "2026-03-02T15:00:00Z", "headline": "Beats", "summary": "", "symbols": ["AAPL"]}],
         "next_page_token": "p2"},
        {"news": [{"created_at": "2026-03-02T16:00:00Z", "headline": "Misses", "summary": "Weak", "symbols": ["AAPL"]}],
         "next_page_token": None},
    ]
    calls = []

    class Resp:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, headers, params, timeout):
        calls.append(dict(params))
        return Resp(pages[len(calls) - 1])

    articles = AlpacaNewsClient(http_get=fake_get).fetch("AAPL", NOW - timedelta(days=1), NOW)

    assert [a.headline for a in articles] == ["Beats", "Misses"]
    assert articles[0].published_at == datetime(2026, 3, 2, 15, 0, tzinfo=timezone.utc)
    assert calls[1]["page_token"] == "p2"
    assert calls[0]["symbols"] == "AAPL"


def test_live_sentiment_reports_unavailable_on_fetch_failure() -> None:
    class BrokenClient:
        def fetch(self, *args, **kwargs):
            raise ConnectionError("down")

    snap = live_sentiment("AAPL", client=BrokenClient(), scorer=LexiconScorer(), now=NOW)

    assert snap.available is False
    features = snap.as_features()
    assert features["sentiment_available"] == 0.0
    assert features["sentiment_score"] != features["sentiment_score"]  # NaN


def test_score_articles_skips_empty_text() -> None:
    articles = [NewsArticle(NOW, "Profit surges"), NewsArticle(NOW, "")]

    scored = score_articles(articles, LexiconScorer())

    assert len(scored) == 1 and scored[0].score > 0
