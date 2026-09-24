#!/usr/bin/env python3
"""
Edge gate: the bot only opens live positions for a setup that has shown an
edge out of sample.

Most automated day-trading strategies lose money after costs. The single
most effective protection is refusing to trade a configuration until a
walk-forward backtest (every traded bar unseen by the model that traded it,
with slippage, spread and the live risk rules) clears minimum bars:

  EDGE_MIN_TRADES             100   enough trades for the stats to mean anything
  EDGE_MIN_FOLDS              3     tested across several periods, not one lucky stretch
  EDGE_MIN_PROFIT_FACTOR      1.2   gross wins / gross losses after costs
  EDGE_MIN_AVG_R              0.05  average profit per trade in units of initial risk
  EDGE_MIN_POSITIVE_FOLDS     0.6   share of folds with positive average R
  EDGE_MAX_DRAWDOWN_PCT       15    worst fold's equity drawdown
  EDGE_REPORT_MAX_AGE_DAYS    30    markets change; re-validate monthly

``scripts/backtest.py --mode walkforward --folds N --promote`` evaluates
these and writes the verdict to EDGE_REPORT_PATH (reports/edge_report.json).
The report records the setup it tested (timeframe, label/ATR parameters,
feature set, threshold and entry gates). ``bot.py --execute`` compares that
with the live setup and blocks new entries (exits are still managed) unless
a fresh, passing, matching report exists. EDGE_GATE=false bypasses it.

Passing is necessary, not sufficient: if you try many settings until one
passes, the winner's result is inflated by selection. Keep the number of
variants you test small and re-validate on newer data before trusting it.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_REPORT_PATH = "reports/edge_report.json"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class EdgeCriteria:
    min_trades: int = 100
    min_folds: int = 3
    min_profit_factor: float = 1.2
    min_avg_r: float = 0.05
    min_positive_fold_share: float = 0.6
    max_drawdown_pct: float = 15.0
    max_report_age_days: float = 30.0

    @classmethod
    def from_env(cls) -> "EdgeCriteria":
        return cls(
            min_trades=int(_env_float("EDGE_MIN_TRADES", cls.min_trades)),
            min_folds=int(_env_float("EDGE_MIN_FOLDS", cls.min_folds)),
            min_profit_factor=_env_float("EDGE_MIN_PROFIT_FACTOR", cls.min_profit_factor),
            min_avg_r=_env_float("EDGE_MIN_AVG_R", cls.min_avg_r),
            min_positive_fold_share=_env_float("EDGE_MIN_POSITIVE_FOLDS", cls.min_positive_fold_share),
            max_drawdown_pct=_env_float("EDGE_MAX_DRAWDOWN_PCT", cls.max_drawdown_pct),
            max_report_age_days=_env_float("EDGE_REPORT_MAX_AGE_DAYS", cls.max_report_age_days),
        )


@dataclass
class EdgeVerdict:
    passed: bool
    failures: List[str]
    metrics: Dict[str, Any]
    setup: Dict[str, Any] = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    criteria: Dict[str, Any] = field(default_factory=dict)


def _num(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value


def _count(value: Any) -> int:
    number = _num(value)
    return int(number) if math.isfinite(number) else 0


def evaluate_edge(metrics: Dict[str, Any], criteria: Optional[EdgeCriteria] = None) -> List[str]:
    """Reasons the pooled walk-forward metrics fail the criteria (empty = pass)."""
    c = criteria or EdgeCriteria.from_env()
    failures = []
    trades = _count(metrics.get("trades"))
    folds = _count(metrics.get("folds"))
    pf = _num(metrics.get("profit_factor"))
    if math.isnan(pf) and trades and _num(metrics.get("win_rate_pct")) == 100:
        pf = math.inf  # no losing trades at all
    avg_r = _num(metrics.get("avg_r"))
    positive = _count(metrics.get("positive_folds"))
    drawdown = abs(_num(metrics.get("max_drawdown_pct")))

    if trades < c.min_trades:
        failures.append(f"{trades} trades < {c.min_trades} required")
    if folds < c.min_folds:
        failures.append(f"{folds} walk-forward folds < {c.min_folds} required (use --folds)")
    if not pf >= c.min_profit_factor:
        failures.append(f"profit factor {pf} < {c.min_profit_factor}")
    if not avg_r >= c.min_avg_r:
        failures.append(f"average R {avg_r} < {c.min_avg_r}")
    if folds and not positive / folds >= c.min_positive_fold_share:
        failures.append(f"only {positive}/{folds} folds had positive average R "
                        f"(need {c.min_positive_fold_share:.0%})")
    if not drawdown <= c.max_drawdown_pct:
        failures.append(f"max drawdown {drawdown}% > {c.max_drawdown_pct}%")
    return failures


def setup_fingerprint(
    *,
    timeframe: str,
    mode: str,
    label_params: Dict[str, Any],
    feature_columns: List[str],
    threshold: float,
    regime_policy: str,
    mtf_confirmation: bool,
    market_filter: bool,
) -> Dict[str, Any]:
    """The parts of a configuration that change which trades are taken."""
    return {
        "timeframe": timeframe,
        "mode": mode,
        "stop_atr_mult": float(label_params.get("stop_atr_mult") or 0),
        "target_atr_mult": float(label_params.get("target_atr_mult") or 0),
        "horizon": int(label_params.get("horizon") or 0) if mode == "model" else None,
        "feature_columns": sorted(feature_columns) if mode == "model" else None,
        "threshold": round(float(threshold), 4),
        "regime_policy": regime_policy,
        "mtf_confirmation": bool(mtf_confirmation),
        "market_filter": bool(market_filter),
    }


def strategy_fingerprint(strategy: Any, timeframe: str) -> Dict[str, Any]:
    """Fingerprint of a live MLStrategy (core.ml_strategy)."""
    labels = dict((strategy.artifact or {}).get("label_params") or {})
    labels.setdefault("stop_atr_mult", strategy.stop_atr_mult)
    labels.setdefault("target_atr_mult", strategy.target_atr_mult)
    return setup_fingerprint(
        timeframe=timeframe,
        mode=strategy.mode,
        label_params=labels,
        feature_columns=list(strategy.feature_columns),
        threshold=strategy.threshold,
        regime_policy=strategy.regime_policy,
        mtf_confirmation=strategy.mtf_confirmation,
        market_filter=getattr(strategy, "market_filter", False),
    )


def build_verdict(
    metrics: Dict[str, Any],
    setup: Dict[str, Any],
    symbols: List[str],
    criteria: Optional[EdgeCriteria] = None,
) -> EdgeVerdict:
    criteria = criteria or EdgeCriteria.from_env()
    failures = evaluate_edge(metrics, criteria)
    clean = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in metrics.items()}
    return EdgeVerdict(not failures, failures, clean, setup, sorted(symbols), criteria=asdict(criteria))


def report_path() -> Path:
    return Path(os.getenv("EDGE_REPORT_PATH", DEFAULT_REPORT_PATH))


def write_report(verdict: EdgeVerdict, path: Optional[Path] = None) -> Path:
    path = Path(path or report_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(verdict), indent=2, default=str))
    return path


def check_live_setup(
    live_setup: Dict[str, Any],
    symbols: List[str],
    *,
    path: Optional[Path] = None,
    criteria: Optional[EdgeCriteria] = None,
    now: Optional[datetime] = None,
) -> tuple[Optional[str], List[str]]:
    """Return (block_reason or None, warnings) for trading ``live_setup`` live."""
    criteria = criteria or EdgeCriteria.from_env()
    path = Path(path or report_path())
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError):
        return (f"no edge report at {path}: run scripts/backtest.py --mode walkforward --folds 4 --promote "
                "with this configuration first"), []
    if not report.get("passed"):
        return f"edge report {path} did not pass: {'; '.join(report.get('failures') or ['unknown'])}", []
    try:
        created = datetime.fromisoformat(str(report.get("created_at")))
    except ValueError:
        return f"edge report {path} has no valid timestamp", []
    now = now or datetime.now(timezone.utc)
    age = now - created
    if age > timedelta(days=criteria.max_report_age_days):
        return (f"edge report is {age.days} days old (max {criteria.max_report_age_days:g}); "
                "re-run the walk-forward on recent data"), []
    tested = report.get("setup") or {}
    diffs = [f"{k}: tested {tested.get(k)!r}, live {v!r}" for k, v in live_setup.items() if tested.get(k) != v]
    if diffs:
        return "live setup differs from the validated one (" + "; ".join(diffs) + ")", []
    untested = sorted(set(symbols) - set(report.get("symbols") or []))
    warnings = [f"symbols not in the validated set: {', '.join(untested)}"] if untested else []
    return None, warnings
