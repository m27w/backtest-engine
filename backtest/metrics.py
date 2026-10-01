"""Performance and risk metrics computed from an equity curve and trade list."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .engine import BacktestResult


@dataclass(frozen=True)
class Metrics:
    total_return_pct: float
    cagr_pct: float
    annual_volatility_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    max_drawdown_bars: int
    calmar: float
    trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    exposure_pct: float
    commission_paid: float

    def as_dict(self) -> dict:
        return asdict(self)


def returns(equity: list[float]) -> list[float]:
    return [b / a - 1 for a, b in zip(equity, equity[1:]) if a != 0]


def max_drawdown(equity: list[float]) -> tuple[float, int]:
    """Largest peak-to-trough fall (as a positive fraction) and the longest
    number of bars spent below a previous peak."""
    peak = equity[0]
    worst = 0.0
    longest = current = 0
    for value in equity:
        if value >= peak:
            peak = value
            current = 0
        else:
            current += 1
            longest = max(longest, current)
            worst = max(worst, (peak - value) / peak)
    return worst, longest


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def compute(result: BacktestResult, periods_per_year: int = 252) -> Metrics:
    equity = [value for _, value in result.equity_curve]
    start, end = equity[0], equity[-1]
    rets = returns(equity)
    years = max(len(equity) - 1, 1) / periods_per_year

    total_return = end / start - 1
    cagr = (end / start) ** (1 / years) - 1 if end > 0 else -1.0
    vol = _stdev(rets) * math.sqrt(periods_per_year)
    mean_ret = _mean(rets)
    sd = _stdev(rets)
    sharpe = mean_ret / sd * math.sqrt(periods_per_year) if sd > 0 else 0.0
    # Sortino only penalises downside volatility.
    downside = math.sqrt(_mean([min(r, 0.0) ** 2 for r in rets])) if rets else 0.0
    sortino = mean_ret / downside * math.sqrt(periods_per_year) if downside > 0 else 0.0
    mdd, mdd_bars = max_drawdown(equity)
    calmar = cagr / mdd if mdd > 0 else 0.0

    pnls = [t.pnl for t in result.trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_loss = -sum(losses)
    profit_factor = sum(wins) / gross_loss if gross_loss > 0 else (math.inf if wins else 0.0)

    return Metrics(
        total_return_pct=total_return * 100,
        cagr_pct=cagr * 100,
        annual_volatility_pct=vol * 100,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=mdd * 100,
        max_drawdown_bars=mdd_bars,
        calmar=calmar,
        trades=len(pnls),
        win_rate_pct=len(wins) / len(pnls) * 100 if pnls else 0.0,
        profit_factor=profit_factor,
        expectancy=_mean(pnls),
        avg_win=_mean(wins),
        avg_loss=_mean(losses),
        exposure_pct=result.bars_in_market / result.total_bars * 100 if result.total_bars else 0.0,
        commission_paid=result.commission_paid,
    )


def format_report(metrics: Metrics, title: str = "Backtest") -> str:
    rows = [
        ("Total return", f"{metrics.total_return_pct:,.2f}%"),
        ("CAGR", f"{metrics.cagr_pct:,.2f}%"),
        ("Annual volatility", f"{metrics.annual_volatility_pct:,.2f}%"),
        ("Sharpe ratio", f"{metrics.sharpe:.2f}"),
        ("Sortino ratio", f"{metrics.sortino:.2f}"),
        ("Max drawdown", f"{metrics.max_drawdown_pct:,.2f}% ({metrics.max_drawdown_bars} bars)"),
        ("Calmar ratio", f"{metrics.calmar:.2f}"),
        ("Trades", f"{metrics.trades}"),
        ("Win rate", f"{metrics.win_rate_pct:.1f}%"),
        ("Profit factor", "inf" if math.isinf(metrics.profit_factor) else f"{metrics.profit_factor:.2f}"),
        ("Expectancy / trade", f"{metrics.expectancy:,.2f}"),
        ("Exposure", f"{metrics.exposure_pct:.1f}%"),
        ("Commission paid", f"{metrics.commission_paid:,.2f}"),
    ]
    width = max(len(k) for k, _ in rows)
    lines = [title, "=" * len(title)]
    lines += [f"{k.ljust(width)}  {v}" for k, v in rows]
    return "\n".join(lines)
