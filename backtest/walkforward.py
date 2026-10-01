"""Parameter optimisation and walk-forward analysis.

Optimising parameters over a whole history and reporting the result is the
most common way backtests lie: the parameters have "seen" the data they are
scored on. Walk-forward analysis guards against that:

    |---- train ----|- test -|
              |---- train ----|- test -|
                        |---- train ----|- test -|

Parameters are chosen on each training window only, then applied unchanged
to the following unseen test window. Only test-window performance is
reported. The gap between in-sample and out-of-sample Sharpe is a direct
measure of overfitting.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

from .data import Bar
from .engine import Backtest
from .metrics import Metrics, compute
from .strategy import Strategy

StrategyFactory = Callable[..., Strategy]


def grid(**param_lists) -> list[dict]:
    """Cartesian product of parameter values: grid(a=[1, 2], b=[3]) -> [{a:1,b:3}, {a:2,b:3}]."""
    keys = list(param_lists)
    return [dict(zip(keys, values)) for values in itertools.product(*param_lists.values())]


def optimise(
    bars: list[Bar],
    factory: StrategyFactory,
    param_grid: list[dict],
    objective: str = "sharpe",
    min_trades: int = 5,
    **backtest_kwargs,
) -> tuple[dict, Metrics]:
    """Return the parameter set with the best objective on ``bars``.

    Parameter sets that produce fewer than ``min_trades`` trades are skipped,
    since a high Sharpe from two trades is noise, not signal.
    """
    best: tuple[dict, Metrics] | None = None
    for params in param_grid:
        try:
            strategy = factory(**params)
        except ValueError:
            continue  # invalid combination, e.g. fast >= slow
        metrics = compute(Backtest(bars, strategy, **backtest_kwargs).run())
        if metrics.trades < min_trades:
            continue
        if best is None or getattr(metrics, objective) > getattr(best[1], objective):
            best = (params, metrics)
    if best is None:
        raise ValueError("No parameter set produced enough trades")
    return best


@dataclass(frozen=True)
class Fold:
    train_range: tuple[int, int]
    test_range: tuple[int, int]
    params: dict
    in_sample: Metrics
    out_of_sample: Metrics


def walk_forward(
    bars: list[Bar],
    factory: StrategyFactory,
    param_grid: list[dict],
    train_size: int = 500,
    test_size: int = 125,
    warmup: int = 100,
    objective: str = "sharpe",
    **backtest_kwargs,
) -> list[Fold]:
    """Roll a train/test window forward through ``bars``.

    ``warmup`` bars before each test window are fed to the strategy so its
    indicators are primed, but trading only starts at the test window itself.
    """
    if warmup > train_size:
        raise ValueError("warmup must not exceed train_size")
    folds: list[Fold] = []
    start = 0
    while start + train_size + test_size <= len(bars):
        train = bars[start: start + train_size]
        params, in_sample = optimise(train, factory, param_grid, objective, **backtest_kwargs)

        test_start = start + train_size
        test_end = test_start + test_size
        window = bars[test_start - warmup: test_end]
        result = Backtest(window, factory(**params), trade_from_index=warmup, **backtest_kwargs).run()

        folds.append(Fold(
            train_range=(start, test_start),
            test_range=(test_start, test_end),
            params=params,
            in_sample=in_sample,
            out_of_sample=compute(result),
        ))
        start += test_size
    return folds
