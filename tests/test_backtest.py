from datetime import datetime, timedelta

import pytest

from backtest import (
    Backtest, Bar, BuyAndHold, Strategy, TrendFollower, compute, grid,
    risk_position_size, synthetic_bars, walk_forward,
)
from backtest.indicators import ATR, EMA, RSI, SMA
from backtest.metrics import max_drawdown


def make_bars(rows):
    """rows: list of (open, high, low, close)."""
    start = datetime(2024, 1, 1)
    return [Bar(start + timedelta(days=i), o, h, l, c) for i, (o, h, l, c) in enumerate(rows)]


class Scripted(Strategy):
    """Submits pre-planned actions on given bar indexes."""

    def __init__(self, actions):
        self.actions = actions
        self.seen_closes = []

    def on_bar(self, ctx):
        self.seen_closes.append(ctx.bar.close)
        action = self.actions.get(ctx.index)
        if action:
            action(ctx)


NO_COSTS = dict(commission_rate=0.0, slippage_bps=0.0)


# ------------------------------------------------------------------ indicators

def test_sma_matches_naive_window():
    xs = [float(i % 7) for i in range(50)]
    sma = SMA(5)
    for i, x in enumerate(xs):
        value = sma.update(x)
        if i < 4:
            assert value is None
        else:
            assert value == pytest.approx(sum(xs[i - 4: i + 1]) / 5)


def test_ema_seeds_with_sma_then_smooths():
    ema = EMA(3)
    assert ema.update(1) is None and ema.update(2) is None
    assert ema.update(3) == pytest.approx(2.0)          # SMA seed
    assert ema.update(5) == pytest.approx(2.0 + 0.5 * (5 - 2.0))


def test_rsi_extremes_and_range():
    up = RSI(5)
    for x in range(1, 20):
        up.update(float(x))
    assert up.value == 100.0
    mixed = RSI(14)
    for x in synthetic_bars(200, seed=3):
        mixed.update(x.close)
    assert 0 < mixed.value < 100


def test_atr_uses_previous_close_for_gaps():
    atr = ATR(2)
    atr.update(10, 9, 10)          # TR = 1
    value = atr.update(15, 14, 15)  # gap up: TR = 15 - 10 = 5
    assert value == pytest.approx(3.0)


# ---------------------------------------------------------------------- engine

def test_orders_fill_at_next_open_not_current_close():
    bars = make_bars([(100, 101, 99, 100), (105, 106, 104, 105), (110, 111, 109, 110)])
    strat = Scripted({0: lambda c: c.buy(1)})
    result = Backtest(bars, strat, **NO_COSTS).run()
    trade = result.trades[0]
    assert trade.entry_price == 105        # bar 1 open, not bar 0 close
    assert trade.exit_reason == "end"
    assert trade.exit_price == 110


def test_strategy_never_sees_future_bars():
    bars = synthetic_bars(100, seed=1)
    strat = Scripted({})
    Backtest(bars, strat).run()
    assert strat.seen_closes == [b.close for b in bars]


def test_stop_assumed_before_target_when_both_hit_same_bar():
    bars = make_bars([
        (100, 100, 100, 100),
        (100, 100, 100, 100),     # fill at 100
        (100, 115, 85, 100),      # both stop (90) and target (110) inside the range
    ])
    strat = Scripted({0: lambda c: c.buy(1, stop_loss=90, take_profit=110)})
    trade = Backtest(bars, strat, **NO_COSTS).run().trades[0]
    assert (trade.exit_reason, trade.exit_price) == ("stop", 90)


def test_gap_through_stop_fills_at_open():
    bars = make_bars([(100, 100, 100, 100), (100, 100, 100, 100), (80, 82, 78, 81)])
    strat = Scripted({0: lambda c: c.buy(1, stop_loss=90)})
    trade = Backtest(bars, strat, **NO_COSTS).run().trades[0]
    assert (trade.exit_reason, trade.exit_price) == ("stop", 80)   # worse than the 90 stop


def test_short_trade_pnl():
    bars = make_bars([(100, 100, 100, 100), (100, 100, 100, 100), (90, 90, 80, 85)])
    strat = Scripted({0: lambda c: c.sell(2, take_profit=85)})
    result = Backtest(bars, strat, **NO_COSTS).run()
    trade = result.trades[0]
    # The short opens at bar 1's open; bar 2 trades down through the 85 target.
    assert trade.entry_price == 100
    assert (trade.exit_reason, trade.exit_price) == ("target", 85)
    assert trade.pnl == pytest.approx(30)
    assert result.final_equity == pytest.approx(10_030)


def test_reversal_closes_then_opens_opposite_side():
    bars = make_bars([(100, 100, 100, 100)] * 2 + [(110, 110, 110, 110)] * 2)
    strat = Scripted({0: lambda c: c.buy(1), 1: lambda c: c.sell(1)})
    result = Backtest(bars, strat, **NO_COSTS).run()
    sides = [(t.side, t.exit_reason) for t in result.trades]
    assert sides == [("long", "signal"), ("short", "end")]


def test_costs_reduce_pnl_by_exact_amount():
    bars = make_bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)])
    strat = Scripted({0: lambda c: c.buy(10)})
    result = Backtest(bars, strat, commission_rate=0.001, slippage_bps=10).run()
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(100.1)   # bought 10 bps worse
    assert trade.exit_price == pytest.approx(99.9)     # sold 10 bps worse
    commission = 10 * 100.1 * 0.001 + 10 * 99.9 * 0.001
    assert trade.pnl == pytest.approx(10 * (99.9 - 100.1) - commission)
    assert result.final_equity == pytest.approx(10_000 + trade.pnl)


def test_warmup_bars_cannot_trade():
    bars = synthetic_bars(50, seed=2)
    strat = Scripted({i: (lambda c: c.buy(1)) for i in range(10)})
    result = Backtest(bars, strat, trade_from_index=10).run()
    assert result.trades == []
    assert len(result.equity_curve) == 40


def test_results_are_deterministic():
    bars = synthetic_bars(800, seed=11)
    a = compute(Backtest(bars, TrendFollower()).run())
    b = compute(Backtest(bars, TrendFollower()).run())
    assert a == b


# --------------------------------------------------------------------- metrics

def test_max_drawdown_and_duration():
    depth, duration = max_drawdown([100, 120, 90, 95, 130, 117])
    assert depth == pytest.approx(0.25)    # 120 -> 90
    assert duration == 2


def test_buy_and_hold_tracks_price():
    bars = synthetic_bars(500, seed=5)
    result = Backtest(bars, BuyAndHold(), **NO_COSTS).run()
    price_return = bars[-1].close / bars[1].open - 1
    metrics = compute(result)
    # 99% invested, so roughly 99% of the price move.
    assert metrics.total_return_pct == pytest.approx(price_return * 99, rel=0.02)
    assert metrics.exposure_pct > 99


def test_risk_position_size():
    assert risk_position_size(10_000, 0.01, 2.0) == pytest.approx(50)
    assert risk_position_size(10_000, 0.01, 0) == 0


# ---------------------------------------------------------------- walk-forward

def test_grid_is_cartesian_product():
    assert grid(a=[1, 2], b=["x"]) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]


def test_walk_forward_windows_do_not_overlap_test_data():
    bars = synthetic_bars(1_200, seed=4)
    folds = walk_forward(
        bars, TrendFollower, grid(fast=[10, 20], slow=[50]),
        train_size=400, test_size=200, warmup=100,
    )
    assert len(folds) == 4
    for fold in folds:
        assert fold.train_range[1] == fold.test_range[0]
    tests = [f.test_range for f in folds]
    assert all(a[1] == b[0] for a, b in zip(tests, tests[1:]))
