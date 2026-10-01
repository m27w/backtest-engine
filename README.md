# backtest-engine

An event-driven backtesting engine for systematic trading strategies, written in pure Python with no dependencies.

It's the research counterpart to my [MQL5 Expert Advisor](https://github.com/m27w/crypto-ea-mt5). Rather than trusting a broker's strategy tester, I wanted an engine where every assumption about fills, costs and timing is explicit, tested and conservative.

```
$ python -m backtest --walk-forward
2500 bars from 2020-01-01 to 2026-11-04

Trend follower (EMA 20/50, RSI filter, 2xATR stop)
==================================================
Total return        -3.90%
Sharpe ratio        -0.14
Max drawdown        12.46% (1274 bars)
Trades              41
Win rate            29.3%
...

Walk-forward analysis (parameters re-optimised on each training window)
fold  params                                         IS Sharpe OOS Sharpe OOS return
   1  fast=10 slow=50 atr=1.5 rr=1.5                      1.11       1.33      3.39%
   2  fast=10 slow=50 atr=1.5 rr=3.0                      1.37      -0.00     -0.05%
   ...
Mean in-sample Sharpe 0.68 vs out-of-sample 0.11 (the gap is the optimisation bias)
```

The demo result is deliberately unflattering. A strategy that looks good after optimisation (in-sample Sharpe 0.68) falls apart on data it hasn't seen (out-of-sample 0.11). Showing that gap honestly is the whole point of the walk-forward module.

## Features

- **Event-driven core.** Bars are processed one at a time; strategies only see closed bars and their orders fill on the next bar, so look-ahead bias is impossible by construction.
- **Realistic execution**: slippage on every fill, per-side commission, ATR-based stop-loss and take-profit brackets, gap handling, long and short positions, and reversals.
- **Conservative intrabar logic.** When a bar's range contains both the stop and the target, the engine assumes the stop was hit first.
- **Streaming indicators** (SMA, EMA, RSI and ATR with Wilder smoothing), each O(1) per bar.
- **Risk-based position sizing**: every trade risks a fixed fraction of equity based on stop distance.
- **Metrics**: CAGR, annualised volatility, Sharpe, Sortino, max drawdown and its duration, Calmar, win rate, profit factor, expectancy and exposure.
- **Grid-search optimisation and walk-forward analysis**, with a minimum-trade filter so parameter sets that "win" on two lucky trades are rejected.
- **Deterministic synthetic data**: a regime-switching geometric Brownian motion, so results are reproducible without licensed market data. Real data loads from CSV.

## Architecture

```
            ┌─────────────┐   on_bar(ctx)   ┌──────────────┐
 bars ────► │  Backtest   │ ──────────────► │   Strategy   │
            │  (engine)   │ ◄────────────── │ (indicators) │
            └─────┬───────┘   buy/sell/close └──────────────┘
                  │ fills at next open, stops/targets intrabar
                  ▼
        equity curve + trades ──► metrics ──► report
                  ▲
   walk-forward: optimise on train window ─► run unchanged on test window
```

| Module | Responsibility |
|---|---|
| `data.py` | `Bar` type with OHLC validation, CSV loader, synthetic price generator |
| `indicators.py` | Streaming SMA, EMA, RSI and ATR |
| `engine.py` | Order handling, fills, slippage, commission, brackets, accounting |
| `strategy.py` | `Strategy` base class, `TrendFollower` and `BuyAndHold` benchmark |
| `metrics.py` | Performance and risk statistics, text report |
| `walkforward.py` | Grid search, optimisation and rolling walk-forward |

## Writing a strategy

```python
from backtest import Backtest, Strategy, compute, format_report, load_csv
from backtest.indicators import SMA

class Breakout(Strategy):
    def __init__(self, period=50):
        self.high = SMA(period)

    def on_bar(self, ctx):
        average = self.high.update(ctx.bar.high)
        if average and ctx.bar.close > average * 1.02 and not ctx.position.is_open:
            ctx.buy(10, stop_loss=ctx.bar.close * 0.95)

result = Backtest(load_csv("btcusd_daily.csv"), Breakout()).run()
print(format_report(compute(result, periods_per_year=365)))
```

## Testing

```bash
pip install pytest
pytest
```

The 18 tests pin down the engine's guarantees:

- orders fill at the **next** open, never the signal bar's close;
- the strategy sees bars strictly in order, never a future bar;
- stop-before-target resolution, gap-through fills, short P&L and reversals;
- commission and slippage reduce P&L by exactly the expected amount;
- warm-up bars can't trade, and walk-forward test windows never overlap;
- indicators match naive reference implementations.

## Limitations

- One net position per strategy (no pyramiding or portfolio of symbols yet).
- Fills are bar-based. Tick-level queue position and partial fills aren't modelled.
- Sharpe uses a zero risk-free rate.

## Licence

MIT
