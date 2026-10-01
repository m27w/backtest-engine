"""Event-driven backtesting engine.

Timeline for every bar ``i``:

1. **Open**: market orders submitted at the previous close are filled at
   this bar's open (plus slippage). A stop or target that the price has
   gapped through is filled at the open, not at the stale level.
2. **Intrabar**: protective stops and take-profits are checked against the
   bar's high and low. If both could have been hit inside the same bar we
   can't know the order, so the engine **assumes the stop was hit first**.
   This is the conservative choice: it never flatters results.
3. **Close**: equity is marked to market, then the strategy's ``on_bar``
   runs and may submit orders, which can only fill at the *next* open.

Because the strategy only ever sees closed bars and its orders only fill on
later bars, look-ahead bias is ruled out by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from .data import Bar

if TYPE_CHECKING:
    from .strategy import Strategy


@dataclass
class Order:
    quantity: float                 # positive = buy/long, negative = sell/short
    stop_loss: float | None = None
    take_profit: float | None = None
    close_only: bool = False


@dataclass
class Position:
    quantity: float = 0.0           # signed
    entry_price: float = 0.0
    entry_time: datetime | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    entry_commission: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.quantity != 0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0


@dataclass(frozen=True)
class Trade:
    side: str                       # "long" or "short"
    quantity: float
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl: float                      # net of commission on both legs
    exit_reason: str                # "signal", "stop", "target" or "end"

    @property
    def return_pct(self) -> float:
        direction = 1 if self.side == "long" else -1
        return direction * (self.exit_price / self.entry_price - 1) * 100


@dataclass
class BacktestResult:
    equity_curve: list[tuple[datetime, float]]
    trades: list[Trade]
    initial_cash: float
    bars_in_market: int
    total_bars: int
    commission_paid: float = 0.0
    params: dict = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1][1]


class Context:
    """What a strategy is allowed to see and do on each bar."""

    def __init__(self, engine: "Backtest"):
        self._engine = engine

    @property
    def bar(self) -> Bar:
        return self._engine.bars[self._engine.index]

    @property
    def index(self) -> int:
        return self._engine.index

    @property
    def position(self) -> Position:
        return self._engine.position

    @property
    def equity(self) -> float:
        return self._engine.equity

    @property
    def cash(self) -> float:
        return self._engine.cash

    def buy(self, quantity: float, stop_loss: float | None = None, take_profit: float | None = None) -> None:
        self._engine.submit(Order(abs(quantity), stop_loss, take_profit))

    def sell(self, quantity: float, stop_loss: float | None = None, take_profit: float | None = None) -> None:
        self._engine.submit(Order(-abs(quantity), stop_loss, take_profit))

    def close(self) -> None:
        self._engine.submit(Order(0.0, close_only=True))


class Backtest:
    def __init__(
        self,
        bars: list[Bar],
        strategy: "Strategy",
        initial_cash: float = 10_000.0,
        commission_rate: float = 0.0005,   # 5 bps of traded notional per side
        slippage_bps: float = 2.0,         # adverse price movement on every fill
        trade_from_index: int = 0,         # bars before this are indicator warm-up only
    ):
        if not bars:
            raise ValueError("No bars to backtest")
        self.bars = bars
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage = slippage_bps / 10_000
        self.trade_from_index = trade_from_index

        self.cash = initial_cash
        self.position = Position()
        self.index = 0
        self.pending: Order | None = None
        self.trades: list[Trade] = []
        self.commission_paid = 0.0

    # ------------------------------------------------------------ state

    @property
    def equity(self) -> float:
        return self.cash + self.position.quantity * self.bars[self.index].close

    def submit(self, order: Order) -> None:
        if self.index < self.trade_from_index:
            return
        if order.quantity == 0 and not order.close_only:
            return
        self.pending = order  # the latest instruction on a bar wins

    # ----------------------------------------------------------- fills

    def _slipped(self, price: float, buying: bool) -> float:
        return price * (1 + self.slippage) if buying else price * (1 - self.slippage)

    def _commission(self, quantity: float, price: float) -> float:
        fee = abs(quantity * price) * self.commission_rate
        self.commission_paid += fee
        return fee

    def _open(self, quantity: float, raw_price: float, bar: Bar, order: Order) -> None:
        price = self._slipped(raw_price, buying=quantity > 0)
        fee = self._commission(quantity, price)
        self.cash -= quantity * price + fee
        self.position = Position(quantity, price, bar.time, order.stop_loss, order.take_profit, fee)

    def _close(self, raw_price: float, bar: Bar, reason: str) -> None:
        pos = self.position
        price = self._slipped(raw_price, buying=pos.quantity < 0)
        fee = self._commission(pos.quantity, price)
        self.cash += pos.quantity * price - fee
        pnl = pos.quantity * (price - pos.entry_price) - pos.entry_commission - fee
        self.trades.append(Trade(
            side="long" if pos.is_long else "short",
            quantity=abs(pos.quantity),
            entry_time=pos.entry_time,  # type: ignore[arg-type]
            exit_time=bar.time,
            entry_price=pos.entry_price,
            exit_price=price,
            pnl=pnl,
            exit_reason=reason,
        ))
        self.position = Position()

    def _fill_pending(self, bar: Bar) -> None:
        order, self.pending = self.pending, None
        if order is None:
            return
        if self.position.is_open:
            reversing = (order.quantity > 0) != self.position.is_long
            if order.close_only or reversing:
                self._close(bar.open, bar, "signal")
            else:
                return  # already positioned in this direction: ignore (no pyramiding)
        if not order.close_only and order.quantity != 0:
            self._open(order.quantity, bar.open, bar, order)

    def _check_exits(self, bar: Bar) -> None:
        pos = self.position
        if not pos.is_open:
            return
        stop, target = pos.stop_loss, pos.take_profit

        if pos.is_long:
            if stop is not None and bar.open <= stop:
                return self._close(bar.open, bar, "stop")        # gapped through stop
            if target is not None and bar.open >= target:
                return self._close(bar.open, bar, "target")      # gapped through target
            if stop is not None and bar.low <= stop:
                return self._close(stop, bar, "stop")             # stop checked first
            if target is not None and bar.high >= target:
                return self._close(target, bar, "target")
        else:
            if stop is not None and bar.open >= stop:
                return self._close(bar.open, bar, "stop")
            if target is not None and bar.open <= target:
                return self._close(bar.open, bar, "target")
            if stop is not None and bar.high >= stop:
                return self._close(stop, bar, "stop")
            if target is not None and bar.low <= target:
                return self._close(target, bar, "target")

    # -------------------------------------------------------------- run

    def run(self) -> BacktestResult:
        context = Context(self)
        self.strategy.on_start(context)
        equity_curve: list[tuple[datetime, float]] = []
        bars_in_market = 0

        for self.index, bar in enumerate(self.bars):
            self._fill_pending(bar)
            self._check_exits(bar)
            if self.position.is_open:
                bars_in_market += 1
            equity_curve.append((bar.time, self.equity))
            self.strategy.on_bar(context)

        if self.position.is_open:
            last = self.bars[-1]
            self._close(last.close, last, "end")
            equity_curve[-1] = (last.time, self.cash)

        return BacktestResult(
            equity_curve=equity_curve[self.trade_from_index:],
            trades=self.trades,
            initial_cash=self.initial_cash,
            bars_in_market=bars_in_market,
            total_bars=len(self.bars) - self.trade_from_index,
            commission_paid=self.commission_paid,
            params=getattr(self.strategy, "params", {}),
        )
