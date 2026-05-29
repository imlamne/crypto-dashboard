import pandas as pd

from src.patterns import detect_all_patterns
from src.price_action import get_price_action_context

SL_PCT = 2.0
MIN_HISTORY = 50
RISK_PCT = 2.0


def _near_support(close: float, support: float | None) -> bool:
    if support is None or support <= 0:
        return False
    return abs(close - support) / support <= 0.01 and close <= support * 1.01


def _at_resistance(row, resistance: float | None) -> bool:
    if resistance is None:
        return False
    return row["high"] >= resistance * 0.995 or row["close"] >= resistance * 0.99


def _has_buy_pattern(patterns: list[dict]) -> bool:
    return any(p["signal"] == "BUY" for p in patterns)


def _check_buy_signal(
    row, prev, history: pd.DataFrame, patterns: list[dict], pa_context: dict
) -> bool:
    rsi = row["rsi"]
    if pd.isna(rsi) or rsi >= 40:
        return False

    ma7 = row["ma_7"]
    prev_ma7 = prev["ma_7"]
    ma7_up = not pd.isna(ma7) and not pd.isna(prev_ma7) and ma7 > prev_ma7

    near_support = _near_support(float(row["close"]), pa_context.get("nearest_support"))
    buy_pattern = _has_buy_pattern(patterns)

    return ma7_up or near_support or buy_pattern


def _check_exit_signal(
    row, prev, entry_idx: int, i: int, entry_price: float, sl_price: float, pa_context: dict
) -> tuple[str | None, float | None]:
    if row["low"] <= sl_price:
        return "SL_HIT", sl_price

    if _at_resistance(row, pa_context.get("nearest_resistance")):
        return "TP_RESISTANCE", float(row["close"])

    rsi = row["rsi"]
    if not pd.isna(rsi) and rsi > 65:
        return "TP_RSI", float(row["close"])

    ma7 = row["ma_7"]
    prev_ma7 = prev["ma_7"]
    if (
        i > entry_idx + 3
        and not pd.isna(ma7)
        and not pd.isna(prev_ma7)
        and ma7 < prev_ma7
    ):
        return "MA_REVERSAL", float(row["close"])

    return None, None


def _trade_result(return_pct: float) -> str:
    if return_pct > 0.05:
        return "WIN"
    if return_pct < -0.05:
        return "LOSS"
    return "BREAKEVEN"


def _close_trade(
    df: pd.DataFrame,
    entry_idx: int,
    exit_idx: int,
    entry_price: float,
    exit_price: float,
    exit_reason: str,
) -> dict:
    return_pct = (exit_price - entry_price) / entry_price * 100
    return {
        "entry_time": df.iloc[entry_idx]["open_time"],
        "entry_price": entry_price,
        "exit_time": df.iloc[exit_idx]["open_time"],
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "return_pct": return_pct,
        "hold_candles": exit_idx - entry_idx,
        "result": _trade_result(return_pct),
    }


def _build_summary(trades: list[dict]) -> dict:
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
            "avg_hold_candles": 0.0,
        }

    returns = [t["return_pct"] for t in trades]
    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]

    equity_curve = get_equity_curve(trades)
    peak = equity_curve[0][1]
    max_dd = 0.0
    for _, equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)

    total_return_pct = (equity_curve[-1][1] / equity_curve[0][1] - 1) * 100

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(trades) * 100,
        "avg_return_pct": sum(returns) / len(returns),
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd,
        "best_trade_pct": max(returns),
        "worst_trade_pct": min(returns),
        "avg_hold_candles": sum(t["hold_candles"] for t in trades) / len(trades),
    }


def run_backtest(df: pd.DataFrame, symbol: str, interval: str) -> dict:
    """
    Run backtest on enriched DataFrame (with indicators).

    Returns dict with trades list and summary statistics.
    """
    trades: list[dict] = []
    if len(df) < MIN_HISTORY + 1:
        return {"trades": trades, "summary": _build_summary(trades), "symbol": symbol, "interval": interval}

    in_trade = False
    entry_price = None
    entry_idx = None
    sl_price = None

    for i in range(MIN_HISTORY, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        history = df.iloc[: i + 1]

        if not in_trade:
            pa_context = get_price_action_context(history)
            patterns = detect_all_patterns(history)

            if _check_buy_signal(row, prev, history, patterns, pa_context):
                in_trade = True
                entry_price = float(row["close"])
                entry_idx = i
                sl_price = entry_price * (1 - SL_PCT / 100)
            continue

        pa_context = get_price_action_context(history)
        exit_reason, exit_price = _check_exit_signal(
            row, prev, entry_idx, i, entry_price, sl_price, pa_context
        )

        if exit_reason:
            trades.append(
                _close_trade(df, entry_idx, i, entry_price, exit_price, exit_reason)
            )
            in_trade = False
            entry_price = None
            entry_idx = None
            sl_price = None

    if in_trade and entry_idx is not None:
        last_idx = len(df) - 1
        last_close = float(df.iloc[last_idx]["close"])
        trades.append(
            _close_trade(df, entry_idx, last_idx, entry_price, last_close, "END_OF_DATA")
        )

    return {
        "trades": trades,
        "summary": _build_summary(trades),
        "symbol": symbol,
        "interval": interval,
    }


def get_equity_curve(
    trades: list[dict], initial_capital: float = 1000
) -> list[tuple[int, float]]:
    """
    Compute equity curve from trades.
    Each trade risks 2% of current capital (SL distance = 2%).
    """
    equity = initial_capital
    curve: list[tuple[int, float]] = [(0, equity)]

    for n, trade in enumerate(trades, start=1):
        # Scale P&L: -2% trade return = -2% of account at 2% risk sizing
        equity *= 1 + (trade["return_pct"] / 100) * (RISK_PCT / SL_PCT)
        curve.append((n, equity))

    return curve
