"""backtest: an event-driven backtesting engine for systematic trading strategies."""

from .data import Bar, load_csv, synthetic_bars
from .engine import Backtest, BacktestResult, Context, Trade
from .metrics import Metrics, compute, format_report
from .strategy import BuyAndHold, Strategy, TrendFollower, risk_position_size
from .walkforward import grid, optimise, walk_forward

__all__ = [
    "Bar", "load_csv", "synthetic_bars",
    "Backtest", "BacktestResult", "Context", "Trade",
    "Metrics", "compute", "format_report",
    "Strategy", "TrendFollower", "BuyAndHold", "risk_position_size",
    "grid", "optimise", "walk_forward",
]
__version__ = "0.1.0"
