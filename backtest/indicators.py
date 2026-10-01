"""Streaming technical indicators.

Each indicator is updated one bar at a time in O(1) and exposes ``value``
(None until it has enough data). Streaming is what makes look-ahead bias
impossible: an indicator can only ever see bars that have already closed.
"""

from __future__ import annotations

from collections import deque


class SMA:
    """Simple moving average using a running sum over a fixed window."""

    def __init__(self, period: int):
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self._window: deque[float] = deque()
        self._sum = 0.0
        self.value: float | None = None

    def update(self, x: float) -> float | None:
        self._window.append(x)
        self._sum += x
        if len(self._window) > self.period:
            self._sum -= self._window.popleft()
        if len(self._window) == self.period:
            self.value = self._sum / self.period
        return self.value


class EMA:
    """Exponential moving average, seeded with the SMA of the first ``period`` values."""

    def __init__(self, period: int):
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.alpha = 2.0 / (period + 1)
        self._seed = SMA(period)
        self.value: float | None = None

    def update(self, x: float) -> float | None:
        if self.value is None:
            self.value = self._seed.update(x)
        else:
            self.value += self.alpha * (x - self.value)
        return self.value


class RSI:
    """Relative Strength Index using Wilder's smoothing (as in most charting platforms)."""

    def __init__(self, period: int = 14):
        self.period = period
        self._previous: float | None = None
        self._gains: list[float] = []
        self._losses: list[float] = []
        self._avg_gain: float | None = None
        self._avg_loss: float | None = None
        self.value: float | None = None

    def update(self, close: float) -> float | None:
        if self._previous is None:
            self._previous = close
            return None
        change = close - self._previous
        self._previous = close
        gain, loss = max(change, 0.0), max(-change, 0.0)

        if self._avg_gain is None:
            self._gains.append(gain)
            self._losses.append(loss)
            if len(self._gains) < self.period:
                return None
            self._avg_gain = sum(self._gains) / self.period
            self._avg_loss = sum(self._losses) / self.period
        else:
            n = self.period
            self._avg_gain = (self._avg_gain * (n - 1) + gain) / n
            self._avg_loss = (self._avg_loss * (n - 1) + loss) / n  # type: ignore[operator]

        if self._avg_loss == 0:
            self.value = 100.0 if self._avg_gain > 0 else 50.0
        else:
            rs = self._avg_gain / self._avg_loss
            self.value = 100.0 - 100.0 / (1.0 + rs)
        return self.value


class ATR:
    """Average True Range with Wilder's smoothing: a volatility measure in price units."""

    def __init__(self, period: int = 14):
        self.period = period
        self._previous_close: float | None = None
        self._seed: list[float] = []
        self.value: float | None = None

    def update(self, high: float, low: float, close: float) -> float | None:
        if self._previous_close is None:
            true_range = high - low
        else:
            true_range = max(
                high - low,
                abs(high - self._previous_close),
                abs(low - self._previous_close),
            )
        self._previous_close = close

        if self.value is None:
            self._seed.append(true_range)
            if len(self._seed) == self.period:
                self.value = sum(self._seed) / self.period
        else:
            self.value = (self.value * (self.period - 1) + true_range) / self.period
        return self.value
