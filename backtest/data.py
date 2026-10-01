"""Market data: the Bar type, CSV loading and a synthetic price generator."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class Bar:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if not (self.low <= min(self.open, self.close) and self.high >= max(self.open, self.close)):
            raise ValueError(f"Inconsistent OHLC values at {self.time}")


def load_csv(path: str | Path) -> list[Bar]:
    """Load bars from a CSV with columns: time,open,high,low,close[,volume].

    ``time`` must be ISO-8601 (e.g. 2024-01-31 or 2024-01-31T14:05:00).
    Rows are sorted by time and duplicate timestamps are rejected.
    """
    bars: list[Bar] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            bars.append(Bar(
                time=datetime.fromisoformat(row["time"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume") or 0.0),
            ))
    bars.sort(key=lambda b: b.time)
    for previous, current in zip(bars, bars[1:]):
        if previous.time == current.time:
            raise ValueError(f"Duplicate timestamp {current.time}")
    return bars


def synthetic_bars(
    n: int = 2_000,
    start_price: float = 100.0,
    seed: int = 0,
    drift: float = 0.0002,
    volatility: float = 0.012,
    regime_length: int = 250,
    start: datetime = datetime(2020, 1, 1),
) -> list[Bar]:
    """Generate daily bars from a regime-switching geometric Brownian motion.

    Every ``regime_length`` bars the drift flips sign with probability 1/2,
    which produces trending and mean-reverting stretches, a more realistic
    test bed for trend-following strategies than a single random walk.
    Deterministic for a given seed.
    """
    rng = random.Random(seed)
    bars: list[Bar] = []
    price = start_price
    mu = drift
    for i in range(n):
        if i and i % regime_length == 0 and rng.random() < 0.5:
            mu = -mu
        # Log-normal step keeps prices positive.
        ret = (mu - 0.5 * volatility ** 2) + volatility * rng.gauss(0.0, 1.0)
        close = price * math.exp(ret)
        open_ = price * math.exp(volatility * 0.25 * rng.gauss(0.0, 1.0))
        spread = abs(rng.gauss(0.0, volatility * 0.6))
        high = max(open_, close) * (1 + spread)
        low = min(open_, close) * (1 - spread)
        bars.append(Bar(start + timedelta(days=i), open_, high, low, close, rng.uniform(1e3, 1e4)))
        price = close
    return bars
