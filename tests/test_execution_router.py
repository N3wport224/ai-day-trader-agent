"""Smart limit chaser: re-peg, slippage abort, fill races, bracket re-anchoring, partial fills."""
from __future__ import annotations

from dataclasses import replace

import pytest
import requests

from core.alpaca_executor import AlpacaExecutor, ExecutionConfig
from core.execution_router import ChaseConfig, SmartLimitChaser
from core.execution_telemetry import EventLog
from core.risk_manager import RiskLimits, RiskManager

CFG = ChaseConfig(buffer_bps=5.0, min_buffer=0.01, fill_timeout_seconds=10, max_slippage_bps=15,
                  max_repegs=3, poll_seconds=1.0)


def _http_error(status=422):
    resp = requests.Response()
    resp.status_code = status
    return requests.exceptions.HTTPError(response=resp)


class FakeBroker:
    """Scripted broker. ``states[order_id]`` is a list of successive get_order
    replies (the last one repeats); quotes are served in order."""

    def __init__(self, states=None, quotes=(), legs=None, replace_fails=False):
        self.states = {k: list(v) for k, v in (states or {}).items()}
        self.quotes = list(quotes)
        self.legs = legs if legs is not None else [
            {"id": "stop-leg", "type": "stop", "status": "held", "stop_price": "98.00"},
            {"id": "tp-leg", "type": "limit", "status": "new", "limit_price": "104.00"},
        ]
        self.replace_fails = replace_fails
        self.open_exits = []   # the account's other open sell orders for the symbol
        self.placed, self.replaced, self.cancelled, self.stop_moves, self.ocos = [], [], [], [], []
        self.next_id = 1

    def place_entry(self, symbol, qty, stop, target, limit_price):
        order_id = f"o{self.next_id}"
        self.next_id += 1
        self.placed.append({"id": order_id, "qty": qty, "stop": stop, "target": target, "limit": limit_price})
        self.states.setdefault(order_id, [{"id": order_id, "status": "new", "filled_qty": "0"}])
        return {"id": order_id, "status": "new", "limit_price": str(limit_price)}

    def get_order(self, order_id, nested=False):
        if nested:
            return {"id": order_id, "legs": self.legs}
        seq = self.states.get(order_id) or [{"id": order_id, "status": "new", "filled_qty": "0"}]
        return seq.pop(0) if len(seq) > 1 else seq[0]

    def replace_order(self, order_id, **fields):
        if order_id in {"stop-leg", "tp-leg"}:
            self.replaced.append((order_id, fields))
            return {"id": order_id}
        if self.replace_fails:
            raise _http_error()
        new_id = f"r{len(self.replaced) + 1}"
        self.replaced.append((order_id, fields))
        self.states.setdefault(new_id, [{"id": new_id, "status": "new", "filled_qty": "0"}])
        return {"id": new_id, "status": "new", "limit_price": fields.get("limit_price")}

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        seq = self.states.get(order_id)
        if seq and str(seq[-1].get("status")) not in {"filled", "canceled"}:
            seq[-1] = {**seq[-1], "status": "canceled"}
        return True

    def quote(self, symbol):
        return self.quotes.pop(0) if self.quotes else None

    def replace_stop_price(self, order_id, stop_price):
        self.stop_moves.append((order_id, stop_price))
        return {"id": order_id}

    def place_exit_oco(self, symbol, qty, stop, target):
        self.ocos.append({"symbol": symbol, "qty": qty, "stop": stop, "target": target})
        return {"id": "oco1"}

    def open_exit_orders(self, symbol):
        return list(self.open_exits)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def _chaser(broker, config=CFG):
    clock = Clock()
    log = EventLog(None)
    return SmartLimitChaser(broker, config, telemetry=log, sleep=clock.sleep, clock=clock), log, clock


def _filled(order_id, price, qty=10):
    return {"id": order_id, "status": "filled", "filled_qty": str(qty), "filled_avg_price": str(price)}


def _events(log, name):
    return [e for e in log.events if e["event"] == name]


def test_limit_is_ask_plus_buffer_capped_by_slippage():
    assert CFG.limit_price(100.00) == 100.05          # 5 bps of $100
    assert CFG.limit_price(10.00) == 10.01            # min buffer $0.01
    assert CFG.slippage_cap(100.00) == 100.15


def test_immediate_fill_reanchors_bracket_to_fill_price():
    broker = FakeBroker(states={"o1": [_filled("o1", 100.03)]})
    chaser, log, _ = _chaser(broker)

    result = chaser.enter("AAPL", 10, stop=98.00, target=104.00, arrival_ask=100.00)

    assert result.status == "filled" and result.fill_price == 100.03 and result.repegs == 0
    assert broker.placed[0]["limit"] == 100.05
    # Planned distances 2.00 / 4.00 from the arrival ask, now from the real fill.
    assert (result.stop, result.target) == (98.03, 104.03)
    assert broker.stop_moves == [("stop-leg", 98.03)]
    assert broker.replaced == [("tp-leg", {"limit_price": "104.03"})]
    assert _events(log, "bracket_reanchored")[0]["fill_price"] == 100.03


def test_unfilled_order_is_repegged_to_new_ask_then_fills():
    broker = FakeBroker(states={"r1": [{"id": "r1", "status": "new", "filled_qty": "0"}, _filled("r1", 100.10)]},
                        quotes=[{"bid": 100.06, "ask": 100.08}])
    chaser, log, clock = _chaser(broker)

    result = chaser.enter("AAPL", 10, 98.00, 104.00, arrival_ask=100.00)

    assert result.status == "filled" and result.order_id == "r1" and result.repegs == 1
    assert broker.replaced[0] == ("o1", {"limit_price": "100.13"})   # 100.08 + 5 bps
    assert clock.t >= CFG.fill_timeout_seconds                        # waited the fill timeout first
    assert _events(log, "entry_repegged")[0]["limit_price"] == 100.13
    assert result.stop == 98.10 and result.target == 104.10


def test_repeg_never_raises_limit_past_the_slippage_cap():
    broker = FakeBroker(states={"r1": [_filled("r1", 100.15)]}, quotes=[{"bid": 100.12, "ask": 100.14}])
    chaser, _, _ = _chaser(broker)
    chaser.enter("AAPL", 10, 98.00, 104.00, arrival_ask=100.00)
    assert broker.replaced[0][1]["limit_price"] == "100.15"   # 100.14 + buffer would be 100.19


def test_runaway_price_cancels_with_slippage_timeout():
    broker = FakeBroker(quotes=[{"bid": 100.20, "ask": 100.25}])   # 25 bps > 15 bps cap
    chaser, log, _ = _chaser(broker)

    result = chaser.enter("AAPL", 10, 98.00, 104.00, arrival_ask=100.00)

    assert result.status == "slippage_timeout" and not result.filled
    assert broker.cancelled == ["o1"] and not broker.replaced
    event = _events(log, "slippage_timeout")[0]
    assert event["ask"] == 100.25 and event["arrival_ask"] == 100.00 and "15 bps cap" in event["reason"]


def test_gives_up_after_max_repegs():
    broker = FakeBroker(quotes=[{"bid": 99.98, "ask": 100.00}] * 5)
    chaser, log, _ = _chaser(broker, ChaseConfig(fill_timeout_seconds=10, max_repegs=2, max_slippage_bps=15))

    result = chaser.enter("AAPL", 10, 98.00, 104.00, arrival_ask=100.00)

    assert result.status == "entry_unfilled" and result.repegs == 2
    assert broker.cancelled == ["o1"]
    assert "2 re-pegs" in _events(log, "entry_unfilled")[0]["reason"]


def test_missing_quote_cancels_instead_of_leaving_order_unsupervised():
    broker = FakeBroker(quotes=[])
    chaser, _, _ = _chaser(broker)
    result = chaser.enter("AAPL", 10, 98.00, 104.00, arrival_ask=100.00)
    assert result.status == "entry_unfilled" and broker.cancelled == ["o1"]


def test_fill_that_races_the_cancel_is_kept_and_protected():
    # Still open at the check, filled by the time the cancel lands.
    broker = FakeBroker(states={"o1": [{"id": "o1", "status": "new", "filled_qty": "0"},
                                       {"id": "o1", "status": "new", "filled_qty": "0"},
                                       _filled("o1", 100.05)]},
                        quotes=[{"bid": 100.25, "ask": 100.30}])
    broker.states["o1"] = [{"id": "o1", "status": "new", "filled_qty": "0"}] * 11 + [_filled("o1", 100.05)]
    chaser, log, _ = _chaser(broker)

    result = chaser.enter("AAPL", 10, 98.00, 104.00, arrival_ask=100.00)

    assert result.status == "filled" and result.fill_price == 100.05
    assert broker.stop_moves == [("stop-leg", 98.05)]
    assert not _events(log, "slippage_timeout")


def test_replace_refused_falls_back_to_cancel_and_resubmit():
    broker = FakeBroker(states={"o2": [_filled("o2", 100.09)]}, quotes=[{"bid": 100.05, "ask": 100.07}],
                        replace_fails=True)
    chaser, log, _ = _chaser(broker)

    result = chaser.enter("AAPL", 10, 98.00, 104.00, arrival_ask=100.00)

    assert broker.cancelled == ["o1"]
    assert broker.placed[1]["limit"] == 100.12 and broker.placed[1]["stop"] == 98.07
    assert result.status == "filled" and result.order_id == "o2"
    assert _events(log, "entry_repegged")[0]["method"] == "cancel_resubmit"


def test_partial_fill_cancels_rest_and_places_oco_when_legs_are_gone():
    broker = FakeBroker(states={"o1": [{"id": "o1", "status": "partially_filled", "filled_qty": "4",
                                        "filled_avg_price": "100.02"}]},
                        legs=[{"id": "stop-leg", "type": "stop", "status": "canceled", "stop_price": "98.00"}])
    chaser, log, _ = _chaser(broker)

    result = chaser.enter("AAPL", 10, 98.00, 104.00, arrival_ask=100.00)

    assert result.status == "partial" and result.filled_qty == 4
    assert broker.cancelled == ["o1"]
    assert broker.ocos == [{"symbol": "AAPL", "qty": 4, "stop": 98.02, "target": 104.02}]
    assert _events(log, "partial_fill_protected")


def test_partial_fill_keeps_working_bracket_legs_without_extra_oco():
    broker = FakeBroker(states={"o1": [{"id": "o1", "status": "partially_filled", "filled_qty": "4",
                                        "filled_avg_price": "100.00"}]})
    chaser, _, _ = _chaser(broker)
    result = chaser.enter("AAPL", 10, 98.00, 104.00, arrival_ask=100.00)
    assert result.status == "partial" and not broker.ocos


def test_failed_oco_on_naked_partial_raises_an_alertable_event():
    broker = FakeBroker(states={"o1": [{"id": "o1", "status": "partially_filled", "filled_qty": "4",
                                        "filled_avg_price": "100.00"}]}, legs=[])

    def boom(*a, **k):
        raise _http_error(403)

    broker.place_exit_oco = boom
    chaser, log, _ = _chaser(broker)
    chaser.enter("AAPL", 10, 98.00, 104.00, arrival_ask=100.00)
    assert _events(log, "unprotected_position")[0]["qty"] == 4


def test_stream_cache_short_circuits_polling():
    broker = FakeBroker()
    calls = []
    original = broker.get_order
    broker.get_order = lambda oid, nested=False: calls.append(oid) or original(oid, nested)
    clock = Clock()
    chaser = SmartLimitChaser(broker, CFG, telemetry=EventLog(None), sleep=clock.sleep, clock=clock,
                              fill_cache=lambda oid: _filled(oid, 100.01))
    result = chaser.enter("AAPL", 10, 98.00, 104.00, arrival_ask=100.00)
    assert result.status == "filled" and clock.t == 0 and calls == ["o1"]   # only the legs read


def test_chase_config_from_env(monkeypatch):
    monkeypatch.setenv("ENTRY_CHASE", "false")
    monkeypatch.setenv("ENTRY_LIMIT_BUFFER_BPS", "3")
    monkeypatch.setenv("ENTRY_FILL_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("ENTRY_MAX_SLIPPAGE_BPS", "20")
    monkeypatch.setenv("ENTRY_MAX_REPEGS", "1")
    cfg = ChaseConfig.from_env()
    assert (cfg.enabled, cfg.buffer_bps, cfg.fill_timeout_seconds, cfg.max_slippage_bps, cfg.max_repegs) == (
        False, 3.0, 5.0, 20.0, 1)


# ---------------------------------------------------------------------------
# Executor integration: BUY signals go through the chaser when a quote exists
# ---------------------------------------------------------------------------

SIGNAL = {"symbol": "AAPL", "recommendation": "BUY", "quantity": 10,
          "risk_parameters": {"stop_distance": 2.0, "target_distance": 4.0}}


@pytest.fixture
def chasing_executor(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")
    for name, value in {"is_market_open": lambda self: True, "get_position": lambda self, s: None,
                        "get_orders_today": lambda self: [],
                        "get_account": lambda self: {"equity": "100000", "last_equity": "100000",
                                                     "buying_power": "100000"}}.items():
        monkeypatch.setattr(AlpacaExecutor, name, value)
    broker = FakeBroker(states={"o1": [_filled("o1", 100.04)]})
    monkeypatch.setattr(AlpacaExecutor, "_place_bracket_order",
                        lambda self, symbol, qty, stop, target, limit_price=None:
                        broker.place_entry(symbol, qty, stop, target, limit_price))
    monkeypatch.setattr(AlpacaExecutor, "get_order", lambda self, oid, nested=False: broker.get_order(oid, nested))
    monkeypatch.setattr(AlpacaExecutor, "replace_order", lambda self, oid, **f: broker.replace_order(oid, **f))
    monkeypatch.setattr(AlpacaExecutor, "replace_stop_price", lambda self, oid, p: broker.replace_stop_price(oid, p))
    monkeypatch.setattr(AlpacaExecutor, "cancel_order", lambda self, oid: broker.cancel_order(oid))
    ex = AlpacaExecutor(
        risk_manager=RiskManager(RiskLimits(min_price=1.0, max_position_pct=1.0, margin_buffer_pct=0)),
        price_lookup=lambda s: 100.0, quote_lookup=lambda s: {"bid": 99.98, "ask": 100.00},
        telemetry=EventLog(None), execution=ExecutionConfig(),
        chase=replace(CFG, fill_timeout_seconds=0),  # real clock here: don't wait out a timeout
    )
    ex.sleep = lambda s: None
    return ex, broker


def test_executor_sends_smart_limit_and_reanchors_on_fill(chasing_executor):
    ex, broker = chasing_executor

    result = ex.submit(SIGNAL)

    assert broker.placed[0]["limit"] == 100.05          # ask 100.00 + 5 bps, not the old 10 bps offset
    assert broker.placed[0]["stop"] == pytest.approx(98.0)
    assert result.chase.status == "filled" and result.order["filled_avg_price"] == "100.04"
    assert broker.stop_moves == [("stop-leg", 98.04)]
    assert ex.expected_prices["o1"] == 100.00          # slippage measured vs the arrival ask


def test_executor_reports_slippage_abort_as_skipped(chasing_executor):
    ex, broker = chasing_executor
    broker.states["o1"] = [{"id": "o1", "status": "new", "filled_qty": "0"}]
    quotes = [{"bid": 99.98, "ask": 100.00}, {"bid": 100.30, "ask": 100.40}]
    ex.quote_lookup = lambda symbol: quotes.pop(0)

    result = ex.submit(SIGNAL)

    assert result.order is None and "slippage_timeout" in result.skipped_reason
    assert broker.cancelled == ["o1"]


def test_fill_without_bracket_legs_gets_an_oco_unless_a_stop_already_exists():
    # E.g. a replaced bracket parent whose legs didn't carry over.
    broker = FakeBroker(states={"o1": [_filled("o1", 100.02)]}, legs=[])
    chaser, _, _ = _chaser(broker)
    chaser.enter("AAPL", 10, 98.00, 104.00, arrival_ask=100.00)
    assert broker.ocos == [{"symbol": "AAPL", "qty": 10, "stop": 98.02, "target": 104.02}]

    covered = FakeBroker(states={"o1": [_filled("o1", 100.02)]}, legs=[])
    covered.open_exits = [{"id": "s9", "type": "stop", "status": "new", "stop_price": "97.50"}]
    chaser, _, _ = _chaser(covered)
    chaser.enter("AAPL", 10, 98.00, 104.00, arrival_ask=100.00)
    assert not covered.ocos and covered.stop_moves == [("s9", 98.02)]   # found and re-anchored instead


def test_network_error_while_repegging_cancels_cleanly():
    broker = FakeBroker(quotes=[{"bid": 100.05, "ask": 100.07}])

    def down(order_id, **fields):
        raise requests.exceptions.ConnectionError("network down")

    broker.replace_order = down
    chaser, log, _ = _chaser(broker)
    result = chaser.enter("AAPL", 10, 98.00, 104.00, arrival_ask=100.00)
    assert result.status == "entry_unfilled" and "re-peg failed" in result.detail
    assert broker.cancelled == ["o1"]


def test_unexpected_chase_error_leaves_the_bracket_working(chasing_executor, monkeypatch):
    ex, broker = chasing_executor
    monkeypatch.setattr(SmartLimitChaser, "enter", lambda *a, **k: 1 / 0)
    result = ex.submit(SIGNAL)
    assert result.order["id"] == "o1" and not broker.cancelled
    assert [e for e in ex.telemetry.events if e["event"] == "entry_chase_error"]
