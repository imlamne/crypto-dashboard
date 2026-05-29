import config
from src.analysis import add_all_indicators
from src.database import get_candles


def _score_to_result(score: int, reasons: list[str]) -> dict:
    if score >= 3:
        return {"score": score, "label": "STRONG BUY", "emoji": "🟢", "reasons": reasons}
    if score == 2:
        return {"score": score, "label": "WEAK BUY", "emoji": "🟡", "reasons": reasons}
    if score == 1:
        return {"score": score, "label": "NEUTRAL", "emoji": "⚪", "reasons": reasons}
    if score == 0:
        return {"score": score, "label": "NEUTRAL", "emoji": "⚪", "reasons": reasons}
    if score == -1:
        return {"score": score, "label": "NEUTRAL", "emoji": "⚪", "reasons": reasons}
    if score == -2:
        return {"score": score, "label": "WEAK SELL", "emoji": "🟠", "reasons": reasons}
    return {"score": score, "label": "STRONG SELL", "emoji": "🔴", "reasons": reasons}


def get_signal(df) -> dict:
    if df.empty or len(df) < 2:
        return _score_to_result(0, [])

    row = df.iloc[-1]
    score = 0
    reasons: list[str] = []

    rsi = row["rsi"]
    if rsi < 30:
        score += 1
        reasons.append(f"RSI oversold ({rsi:.1f})")
    elif rsi > 70:
        score -= 1
        reasons.append(f"RSI overbought ({rsi:.1f})")

    close = row["close"]
    if close < row["bb_lower"]:
        score += 1
        reasons.append("Price below BB lower")
    elif close > row["bb_upper"]:
        score -= 1
        reasons.append("Price above BB upper")

    if len(df) >= 11:
        avg_volume = df["volume"].iloc[-11:-1].mean()
        current_volume = row["volume"]
        if current_volume < avg_volume:
            score += 1
            reasons.append("Volume decreasing")
        elif current_volume > avg_volume:
            score -= 1
            reasons.append("Volume increasing")

    ma_7_slope = df["ma_7"].diff().iloc[-1]
    if ma_7_slope > 0:
        score += 1
        reasons.append("MA7 sloping up")
    elif ma_7_slope < 0:
        score -= 1
        reasons.append("MA7 sloping down")

    return _score_to_result(score, reasons)


def get_all_signals() -> dict[tuple[str, str], dict]:
    signals: dict[tuple[str, str], dict] = {}

    for symbol in config.TRADING_PAIRS:
        for interval in config.TIMEFRAMES:
            df = get_candles(symbol, interval, config.DEFAULT_CANDLE_LIMIT)
            if df.empty:
                signals[(symbol, interval)] = _score_to_result(0, [])
                continue

            enriched = add_all_indicators(df)
            signals[(symbol, interval)] = get_signal(enriched)

    return signals


def get_mtf_conclusion(symbol: str) -> str:
    all_signals = get_all_signals()
    symbol_signals = {key: value for key, value in all_signals.items() if key[0] == symbol}

    buy_count = sum(1 for signal in symbol_signals.values() if "BUY" in signal["label"])
    sell_count = sum(1 for signal in symbol_signals.values() if "SELL" in signal["label"])

    if buy_count >= 1 and sell_count >= 1:
        return "CONFLICTED"
    if buy_count >= 3:
        return "STRONG BUY"
    if sell_count >= 3:
        return "STRONG SELL"
    if buy_count >= 2:
        return "WEAK BUY"
    if sell_count >= 2:
        return "WEAK SELL"
    return "NEUTRAL"
