"""Paper self-test: safe order path, full round trip, cleanup, API and System check row."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
import requests

from config.api import control
from core.execution_router import ChaseConfig
from core.execution_telemetry import EventLog
from core.health import check_selftest
from core.selftest import SelfTest, run_selftest


def _http_error(status=422, text="refused"):
    resp = requests.Response()
    resp.status_code = status
    resp._content = text.encode()
    return requests.exceptions.HTTPError(response=resp)


class FakeAlpaca:
    """A tiny paper account: orders with bracket legs, replace, cancel, fills."""

    def __init__(self, market_open=True, bid=500.00, ask=500.02, replace="keep_legs", cancel_works=True,
                 fill_price=500.03, account=None):
        self.mode = "paper"
        self.telemetry = EventLog(None)
        self.chase = ChaseConfig(fill_timeout_seconds=0, max_repegs=1, poll_seconds=0)
        self.market_open, self.bid, self.ask = market_open, bid, ask
        self.replace_mode, self.cancel_works, self.fill_price = replace, cancel_works, fill_price
        self.account = account or {"status": "ACTIVE", "equity": "100000", "buying_power": "200000"}
        self.orders, self.calls, self.next_id = {}, [], 1
        self.fill_entries = False  # full test: marketable entries fill at once

    # account / data
    def get_account(self):
        if isinstance(self.account, Exception):
            raise self.account
        return self.account

    def get_clock(self):
        return {"is_open": self.market_open, "next_open": "2026-09-28T09:30:00-04:00"}

    def get_quote(self, symbol):
        return {"bid": self.bid, "ask": self.ask}

    def _quote(self, symbol):
        return self.get_quote(symbol)

    # orders
    def _new(self, **order):
        order_id = f"id{self.next_id}"
        self.next_id += 1
        self.orders[order_id] = {"id": order_id, "status": "new", "filled_qty": "0", **order}
        return self.orders[order_id]

    def _place_bracket_order(self, symbol, qty, stop, target, limit_price=None):
        self.calls.append(("bracket", limit_price))
        legs = [self._new(type="stop", side="sell", stop_price=str(stop), status="held"),
                self._new(type="limit", side="sell", limit_price=str(target))]
        order = self._new(symbol=symbol, side="buy", type="limit", limit_price=str(limit_price), legs=legs)
        if self.fill_entries:
            order.update(status="filled", filled_qty=str(qty), filled_avg_price=str(self.fill_price))
            for leg in legs:
                leg["status"] = "new"
        return dict(order)

    def get_order(self, order_id, nested=False):
        order = dict(self.orders[order_id])
        order["legs"] = [dict(self.orders[leg["id"]]) for leg in order.get("legs") or []] if nested else None
        return order

    def replace_order(self, order_id, **fields):
        self.calls.append(("replace", order_id, fields))
        if self.orders[order_id]["side"] == "sell":          # a leg (re-anchoring)
            self.orders[order_id].update(fields)
            return dict(self.orders[order_id])
        if self.replace_mode == "refuse":
            raise _http_error(422, "replace not supported")
        old = self.orders[order_id]
        old["status"] = "replaced"
        legs = old["legs"] if self.replace_mode == "keep_legs" else []
        new = self._new(symbol=old["symbol"], side="buy", type="limit", legs=legs,
                        limit_price=str(fields.get("limit_price")))
        old["replaced_by"] = new["id"]
        return dict(new)

    def replace_stop_price(self, order_id, stop_price):
        self.calls.append(("stop", order_id, stop_price))
        self.orders[order_id]["stop_price"] = str(stop_price)
        return dict(self.orders[order_id])

    def cancel_order(self, order_id):
        self.calls.append(("cancel", order_id))
        order = self.orders[order_id]
        if self.cancel_works and order["status"] not in {"filled", "canceled"}:
            order["status"] = "canceled"
            for leg in order.get("legs") or []:
                if self.orders[leg["id"]]["status"] in {"held", "new"} and order["side"] == "buy":
                    self.orders[leg["id"]]["status"] = "canceled"
        return True

    def _place_order(self, symbol, qty, side):
        self.calls.append(("market", side, qty))
        return dict(self._new(symbol=symbol, side=side, type="market", status="filled",
                              filled_qty=str(qty), filled_avg_price="500.01"))

    def place_exit_oco(self, symbol, qty, stop, target):
        self.calls.append(("oco", qty))
        return {"id": "oco"}

    def _chase_broker(self):
        from core.alpaca_executor import _ChaseBroker

        return _ChaseBroker(self)

    def working(self):
        return [o for o in self.orders.values() if o["status"] in {"new", "held", "accepted"}]


class FakeStream:
    def __init__(self, status="connected", seen=True, alpaca=None):
        self._status, self.seen, self.alpaca = status, seen, alpaca
        self.failed = "unauthorized" if status == "failed" else None
        self.started = self.stopped = False

    @property
    def status(self):
        return self._status

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def order_update(self, order_id):
        return self.alpaca.get_order(order_id) if self.seen and self.alpaca and order_id in self.alpaca.orders else None


def _run(alpaca, stream="connected", full=False, seen=True):
    streams = []

    def factory(ex):
        streams.append(FakeStream(stream, seen, alpaca))
        return streams[-1]

    result = SelfTest(alpaca, full=full, stream_factory=factory if stream else None,
                      sleep=lambda s: None, stream_wait=1).run()
    return result, {s["id"]: s for s in result["steps"]}, streams


def test_refuses_the_live_account():
    alpaca = FakeAlpaca()
    alpaca.mode = "live"
    with pytest.raises(ValueError, match="paper"):
        run_selftest(alpaca)


def test_market_closed_checks_connections_and_places_no_orders():
    alpaca = FakeAlpaca(market_open=False, bid=0, ask=0)
    result, steps, streams = _run(alpaca)

    assert steps["account"]["status"] == "ok" and steps["stream"]["status"] == "ok"
    assert steps["quote"]["status"] == "info"
    assert steps["order_place"]["status"] == "info" and "market hours" in steps["order_place"]["detail"]
    assert not alpaca.orders and result["overall"] == "ok"
    assert streams[0].started and streams[0].stopped


def test_quick_test_places_reprices_and_cancels_a_non_filling_order():
    alpaca = FakeAlpaca()
    result, steps, _ = _run(alpaca)

    assert alpaca.calls[0] == ("bracket", 475.0)                      # 5% under the $500 bid
    assert steps["order_place"]["status"] == "ok" and "2 protective" in steps["order_place"]["detail"]
    assert steps["order_replace"]["status"] == "ok" and "stayed attached" in steps["order_replace"]["detail"]
    assert steps["order_cancel"]["status"] == "ok"
    assert steps["stream_updates"]["status"] == "ok"
    assert result["overall"] == "ok" and not alpaca.working()            # nothing left behind
    assert not [c for c in alpaca.calls if c[0] == "market"]             # nothing bought or sold


def test_refused_replace_is_a_warning_and_the_order_is_still_cancelled():
    alpaca = FakeAlpaca(replace="refuse")
    result, steps, _ = _run(alpaca)
    assert steps["order_replace"]["status"] == "warn" and "cancels and resubmits" in steps["order_replace"]["fix"]
    assert steps["order_cancel"]["status"] == "ok" and result["overall"] == "warn" and not alpaca.working()


def test_replace_that_drops_the_legs_is_reported():
    alpaca = FakeAlpaca(replace="drop_legs")
    _, steps, _ = _run(alpaca)
    assert steps["order_replace"]["status"] == "warn" and "0 protective" in steps["order_replace"]["detail"]


def test_unconfirmed_cancel_fails_with_instructions():
    alpaca = FakeAlpaca(cancel_works=False)
    result, steps, _ = _run(alpaca)
    assert steps["order_cancel"]["status"] == "fail" and "alpaca.markets" in steps["order_cancel"]["fix"]
    assert result["overall"] == "fail"


def test_bad_keys_stop_before_any_order():
    alpaca = FakeAlpaca(account=_http_error(401, "unauthorized"))
    result, steps, _ = _run(alpaca)
    assert steps["account"]["status"] == "fail" and "API Keys" in steps["account"]["fix"]
    assert list(steps) == ["account"] and not alpaca.orders and result["overall"] == "fail"


def test_network_failure_is_not_blamed_on_the_keys():
    alpaca = FakeAlpaca(account=requests.exceptions.ConnectionError("ProxyError(... 403 Forbidden ...)"))
    _, steps, _ = _run(alpaca)
    assert steps["account"]["detail"] == "Could not reach Alpaca (no connection)"
    assert "firewall" in steps["account"]["fix"] and "API Keys" not in steps["account"]["fix"]


def test_stream_login_refused_or_turned_off():
    _, steps, _ = _run(FakeAlpaca(market_open=False), stream="failed")
    assert steps["stream"]["status"] == "fail" and "unauthorized" in steps["stream"]["detail"]
    _, steps, _ = _run(FakeAlpaca(market_open=False), stream="reconnecting")
    assert steps["stream"]["status"] == "warn" and "WebSockets" in steps["stream"]["fix"]
    _, steps, _ = _run(FakeAlpaca(market_open=False), stream=None)
    assert steps["stream"]["status"] == "info"


def test_full_test_buys_checks_protection_and_sells_after_legs_are_cancelled():
    alpaca = FakeAlpaca()
    alpaca.fill_entries = True
    result, steps, _ = _run(alpaca, full=True)

    assert steps["buy"]["status"] == "ok" and "$500.03" in steps["buy"]["detail"]
    assert steps["protection"]["status"] == "ok"                       # stop moved to fill - planned distance
    assert steps["stream_fill"]["status"] == "ok"
    assert steps["sell"]["status"] == "ok"
    sell_at = alpaca.calls.index(("market", "sell", 1))
    leg_cancels = [i for i, c in enumerate(alpaca.calls) if c[0] == "cancel" and alpaca.orders[c[1]]["side"] == "sell"]
    assert leg_cancels and max(leg_cancels) < sell_at                  # legs released before selling
    assert result["overall"] == "ok" and not alpaca.working()


def test_full_test_never_sells_while_a_leg_might_still_be_working():
    alpaca = FakeAlpaca(cancel_works=False)
    alpaca.fill_entries = True
    _, steps, _ = _run(alpaca, full=True)
    assert steps["sell"]["status"] == "fail" and "still protected" in steps["sell"]["fix"]
    assert ("market", "sell", 1) not in alpaca.calls


def test_unexpected_error_is_reported_and_test_orders_are_cleaned_up():
    alpaca = FakeAlpaca()
    original = alpaca.replace_order

    def boom(order_id, **fields):
        original(order_id, **fields)
        raise RuntimeError("something odd")

    alpaca.replace_order = boom
    result, steps, streams = _run(alpaca)
    assert steps["error"]["status"] == "fail" and result["overall"] == "fail"
    assert not alpaca.working() and streams[0].stopped


def test_selftest_event_reaches_the_activity_feed():
    alpaca = FakeAlpaca(market_open=False)
    _run(alpaca)
    assert [e for e in alpaca.telemetry.events if e["event"] == "selftest"][0]["overall"] == "ok"


# ---------------------------------------------------------------------------
# System check row
# ---------------------------------------------------------------------------

NOW = datetime(2026, 9, 28, 15, 0, tzinfo=timezone.utc)


def _result(overall, order_status="ok", days=0):
    return {"overall": overall, "checked_at": (NOW - timedelta(days=days)).isoformat(),
            "steps": [{"id": "order_cancel", "label": "Cancel order", "status": order_status},
                      {"id": "stream", "label": "Live order stream", "status": "fail" if overall == "fail" else "ok"}]}


def test_system_check_row():
    assert check_selftest({}, NOW).status == "info"
    assert check_selftest(_result("ok"), NOW).status == "ok"
    assert "warnings" in check_selftest(_result("warn"), NOW).detail
    failed = check_selftest(_result("fail"), NOW)
    assert failed.status == "warn" and "Live order stream" in failed.detail
    closed = check_selftest(_result("ok", order_status="info"), NOW)
    assert closed.status == "info" and "market hours" in closed.detail
    assert check_selftest(_result("ok", days=30), NOW).status == "info"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_api_guards_and_runs(client, monkeypatch, manager, tmp_path):
    monkeypatch.setattr(control, "selftest_path", lambda root=None: tmp_path / "selftest.json")
    assert client.post("/api/control/paper/selftest", json={}).status_code == 400        # not confirmed
    assert client.post("/api/control/paper/selftest", json={"confirm": True}).status_code == 400  # no keys

    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    calls = []

    def fake_run(symbol, full):
        calls.append((symbol, full))
        result = {"overall": "ok", "symbol": symbol, "full": full, "steps": [], "checked_at": NOW.isoformat()}
        (tmp_path / "selftest.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(control, "_run_selftest", fake_run)
    resp = client.post("/api/control/paper/selftest", json={"confirm": True, "full": True, "symbol": "qqq"})
    assert resp.status_code == 200 and calls == [("QQQ", True)]
    assert client.get("/api/control/paper/selftest").json()["symbol"] == "QQQ"
    assert client.post("/api/control/paper/selftest",
                       json={"confirm": True, "symbol": "BAD;SYM"}).status_code == 422

    monkeypatch.setattr(type(manager.bots["paper"]), "running", lambda self: True)
    assert client.post("/api/control/paper/selftest", json={"confirm": True}).status_code == 409
