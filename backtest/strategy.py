"""Strategy interface and an example trend-following strategy."""

from __future__ import annotations

from .engine import Context
from .indicators import ATR, EMA, RSI


class Strategy:
    """Subclass and implement ``on_bar``. Runs after each bar closes."""

    params: dict = {}

    def on_start(self, ctx: Context) -> None:  # pragma: no cover - optional hook
        pass

    def on_bar(self, ctx: Context) -> None:
        raise NotImplementedError


def risk_position_size(equity: float, risk_fraction: float, stop_distance: float) -> float:
    """Size a position so that hitting the stop loses ``risk_fraction`` of equity.

    e.g. £10,000 equity, 1% risk, £2 stop distance -> 50 units (50 * £2 = £100).
    """
    if stop_distance <= 0:
        return 0.0
    return (equity * risk_fraction) / stop_distance


class TrendFollower(Strategy):
    """EMA crossover with an RSI filter, ATR stops and fixed-fractional sizing.

    * Long when the fast EMA crosses above the slow EMA and RSI is not overbought.
    * Short (optional) on the opposite cross when RSI is not oversold.
    * Stop = entry -/+ ``atr_mult`` * ATR; target = ``reward_risk`` * stop distance.
    * Each trade risks ``risk_fraction`` of current equity.

    The same structure as my MQL5 expert advisor, rebuilt as a testable
    Python strategy.
    """

    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        atr_period: int = 14,
        atr_mult: float = 2.0,
        reward_risk: float = 2.0,
        risk_fraction: float = 0.01,
        allow_shorts: bool = True,
    ):
        if fast >= slow:
            raise ValueError("fast period must be shorter than slow period")
        self.params = {
            "fast": fast, "slow": slow, "atr_mult": atr_mult,
            "reward_risk": reward_risk, "risk_fraction": risk_fraction,
        }
        self.fast, self.slow = EMA(fast), EMA(slow)
        self.rsi, self.atr = RSI(rsi_period), ATR(atr_period)
        self.rsi_overbought, self.rsi_oversold = rsi_overbought, rsi_oversold
        self.atr_mult, self.reward_risk = atr_mult, reward_risk
        self.risk_fraction, self.allow_shorts = risk_fraction, allow_shorts
        self._previous_diff: float | None = None

    def on_bar(self, ctx: Context) -> None:
        bar = ctx.bar
        fast = self.fast.update(bar.close)
        slow = self.slow.update(bar.close)
        rsi = self.rsi.update(bar.close)
        atr = self.atr.update(bar.high, bar.low, bar.close)
        if None in (fast, slow, rsi, atr):
            return

        diff = fast - slow  # type: ignore[operator]
        previous, self._previous_diff = self._previous_diff, diff
        if previous is None:
            return

        crossed_up = previous <= 0 < diff
        crossed_down = previous >= 0 > diff
        stop_distance = self.atr_mult * atr  # type: ignore[operator]
        size = risk_position_size(ctx.equity, self.risk_fraction, stop_distance)
        # Never use more than 100% of equity as notional (no leverage).
        size = min(size, ctx.equity / bar.close)

        if crossed_up and rsi < self.rsi_overbought:  # type: ignore[operator]
            ctx.buy(
                size,
                stop_loss=bar.close - stop_distance,
                take_profit=bar.close + self.reward_risk * stop_distance,
            )
        elif crossed_down:
            if self.allow_shorts and rsi > self.rsi_oversold:  # type: ignore[operator]
                ctx.sell(
                    size,
                    stop_loss=bar.close + stop_distance,
                    take_profit=bar.close - self.reward_risk * stop_distance,
                )
            elif ctx.position.is_long:
                ctx.close()


class BuyAndHold(Strategy):
    """Benchmark: buy with all equity on the first bar and hold."""

    def on_bar(self, ctx: Context) -> None:
        if not ctx.position.is_open:
            # Leave a small buffer for slippage and commission on the fill.
            ctx.buy(ctx.equity * 0.99 / ctx.bar.close)
