"""Command-line demo: python -m backtest [prices.csv]

Without a CSV it runs on deterministic synthetic data, so results are reproducible.
"""

from __future__ import annotations

import argparse

from . import (
    Backtest, BuyAndHold, TrendFollower, compute, format_report, grid, load_csv,
    synthetic_bars, walk_forward,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest a trend-following strategy.")
    parser.add_argument("csv", nargs="?", help="CSV with time,open,high,low,close[,volume]")
    parser.add_argument("--seed", type=int, default=7, help="seed for synthetic data")
    parser.add_argument("--walk-forward", action="store_true", help="run walk-forward analysis")
    args = parser.parse_args(argv)

    bars = load_csv(args.csv) if args.csv else synthetic_bars(n=2_500, seed=args.seed)
    print(f"{len(bars)} bars from {bars[0].time:%Y-%m-%d} to {bars[-1].time:%Y-%m-%d}\n")

    strategy_result = Backtest(bars, TrendFollower()).run()
    print(format_report(compute(strategy_result), "Trend follower (EMA 20/50, RSI filter, 2xATR stop)"))
    print()
    print(format_report(compute(Backtest(bars, BuyAndHold()).run()), "Buy and hold benchmark"))

    if args.walk_forward:
        param_grid = grid(fast=[10, 20, 30], slow=[50, 100], atr_mult=[1.5, 2.5], reward_risk=[1.5, 3.0])
        folds = walk_forward(bars, TrendFollower, param_grid, train_size=750, test_size=250, warmup=150)
        print("\nWalk-forward analysis (parameters re-optimised on each training window)")
        print(f"{'fold':>4}  {'params':<46} {'IS Sharpe':>9} {'OOS Sharpe':>10} {'OOS return':>10}")
        for i, fold in enumerate(folds, 1):
            p = fold.params
            label = f"fast={p['fast']} slow={p['slow']} atr={p['atr_mult']} rr={p['reward_risk']}"
            print(f"{i:>4}  {label:<46} {fold.in_sample.sharpe:>9.2f} "
                  f"{fold.out_of_sample.sharpe:>10.2f} {fold.out_of_sample.total_return_pct:>9.2f}%")
        is_avg = sum(f.in_sample.sharpe for f in folds) / len(folds)
        oos_avg = sum(f.out_of_sample.sharpe for f in folds) / len(folds)
        print(f"\nMean in-sample Sharpe {is_avg:.2f} vs out-of-sample {oos_avg:.2f} "
              f"(the gap is the optimisation bias)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
