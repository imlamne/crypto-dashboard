import pandas as pd


def _empty_result() -> dict:
    return {
        "detected": False,
        "name": "",
        "emoji": "",
        "signal": "NEUTRAL",
        "description": "",
        "pattern_type": "",
    }


def analyze_candle(row):
    body = abs(row["close"] - row["open"])
    full_range = row["high"] - row["low"]
    upper_wick = row["high"] - max(row["close"], row["open"])
    lower_wick = min(row["close"], row["open"]) - row["low"]
    is_bullish = row["close"] > row["open"]
    is_bearish = row["close"] < row["open"]
    body_pct = body / full_range if full_range > 0 else 0
    return body, full_range, upper_wick, lower_wick, is_bullish, is_bearish, body_pct


def detect_hammer(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return _empty_result()

    row = df.iloc[-1]
    body, full_range, upper_wick, lower_wick, _, _, _ = analyze_candle(row)
    if full_range <= 0 or body <= 0:
        return _empty_result()

    body_bottom = min(row["open"], row["close"])
    body_in_upper_third = body_bottom >= row["low"] + 0.7 * full_range
    prior_bearish = all(df.iloc[i]["close"] < df.iloc[i]["open"] for i in (-3, -2))

    if (
        body_in_upper_third
        and lower_wick >= 2 * body
        and upper_wick <= 0.3 * body
        and prior_bearish
    ):
        return {
            "detected": True,
            "name": "Hammer",
            "emoji": "🔨",
            "signal": "BUY",
            "description": "Búa — seller đẩy giá xuống mạnh nhưng buyer phản công. Tín hiệu đảo chiều tăng.",
            "pattern_type": "hammer",
        }
    return _empty_result()


def detect_shooting_star(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return _empty_result()

    row = df.iloc[-1]
    body, full_range, upper_wick, lower_wick, _, _, _ = analyze_candle(row)
    if full_range <= 0 or body <= 0:
        return _empty_result()

    body_top = max(row["open"], row["close"])
    body_in_lower_third = body_top <= row["low"] + 0.3 * full_range
    prior_bullish = all(df.iloc[i]["close"] > df.iloc[i]["open"] for i in (-3, -2))

    if (
        body_in_lower_third
        and upper_wick >= 2 * body
        and lower_wick <= 0.3 * body
        and prior_bullish
    ):
        return {
            "detected": True,
            "name": "Shooting Star",
            "emoji": "🌠",
            "signal": "SELL",
            "description": "Sao băng — buyer đẩy giá lên cao nhưng seller phản công mạnh. Tín hiệu đảo chiều giảm.",
            "pattern_type": "shooting_star",
        }
    return _empty_result()


def detect_doji(df: pd.DataFrame) -> dict:
    if len(df) < 1:
        return _empty_result()

    row = df.iloc[-1]
    body, full_range, upper_wick, lower_wick, _, _, body_pct = analyze_candle(row)
    if full_range <= 0:
        return _empty_result()

    if body_pct <= 0.05 and upper_wick > 0 and lower_wick > 0:
        return {
            "detected": True,
            "name": "Doji",
            "emoji": "⚖️",
            "signal": "NEUTRAL",
            "description": "Doji — thị trường do dự, buyer và seller cân bằng. Cảnh báo đảo chiều hoặc consolidation.",
            "pattern_type": "doji",
        }
    return _empty_result()


def detect_bullish_engulfing(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        return _empty_result()

    prev = df.iloc[-2]
    curr = df.iloc[-1]
    _, _, _, _, prev_bearish, _, _ = analyze_candle(prev)
    _, _, _, _, curr_bullish, _, _ = analyze_candle(curr)

    if (
        prev_bearish
        and curr_bullish
        and curr["open"] <= prev["close"]
        and curr["close"] >= prev["open"]
    ):
        return {
            "detected": True,
            "name": "Bullish Engulfing",
            "emoji": "🟢",
            "signal": "BUY",
            "description": "Bullish Engulfing — nến xanh nuốt hoàn toàn nến đỏ trước. Buyer áp đảo seller hoàn toàn.",
            "pattern_type": "bullish_engulfing",
        }
    return _empty_result()


def detect_bearish_engulfing(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        return _empty_result()

    prev = df.iloc[-2]
    curr = df.iloc[-1]
    _, _, _, _, prev_bullish, _, _ = analyze_candle(prev)
    _, _, _, _, _, curr_bearish, _ = analyze_candle(curr)

    if (
        prev_bullish
        and curr_bearish
        and curr["open"] >= prev["close"]
        and curr["close"] <= prev["open"]
    ):
        return {
            "detected": True,
            "name": "Bearish Engulfing",
            "emoji": "🔴",
            "signal": "SELL",
            "description": "Bearish Engulfing — nến đỏ nuốt hoàn toàn nến xanh trước. Seller áp đảo buyer hoàn toàn.",
            "pattern_type": "bearish_engulfing",
        }
    return _empty_result()


def detect_pin_bar(df: pd.DataFrame) -> dict:
    if len(df) < 1:
        return _empty_result()

    row = df.iloc[-1]
    body, full_range, upper_wick, lower_wick, _, _, body_pct = analyze_candle(row)
    if full_range <= 0 or body <= 0:
        return _empty_result()

    if body_pct > 0.25:
        return _empty_result()

    if lower_wick >= 3 * body and lower_wick > upper_wick:
        return {
            "detected": True,
            "name": "Bullish Pin Bar",
            "emoji": "📌",
            "signal": "BUY",
            "description": "Pin Bar tăng — giá bị reject mạnh tại vùng thấp. Buyer không chấp nhận giá thấp hơn.",
            "pattern_type": "pin_bar",
        }

    if upper_wick >= 3 * body and upper_wick > lower_wick:
        return {
            "detected": True,
            "name": "Bearish Pin Bar",
            "emoji": "📌",
            "signal": "SELL",
            "description": "Pin Bar giảm — giá bị reject mạnh tại vùng cao. Seller không chấp nhận giá cao hơn.",
            "pattern_type": "pin_bar",
        }
    return _empty_result()


def detect_inside_bar(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        return _empty_result()

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    if curr["high"] < prev["high"] and curr["low"] > prev["low"]:
        return {
            "detected": True,
            "name": "Inside Bar",
            "emoji": "📦",
            "signal": "NEUTRAL",
            "description": "Inside Bar — thị trường đang tích lũy trong nến mẹ. Chờ breakout để xác định hướng.",
            "pattern_type": "inside_bar",
        }
    return _empty_result()


def detect_morning_star(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return _empty_result()

    c1 = df.iloc[-3]
    c2 = df.iloc[-2]
    c3 = df.iloc[-1]

    _, _, _, _, c1_bearish, _, c1_body_pct = analyze_candle(c1)
    _, _, _, _, _, _, c2_body_pct = analyze_candle(c2)
    _, _, _, _, c3_bullish, _, c3_body_pct = analyze_candle(c3)

    c1_mid = (c1["open"] + c1["close"]) / 2
    gaps_down = max(c2["open"], c2["close"]) < c1["close"]

    if (
        c1_bearish
        and c1_body_pct >= 0.5
        and c2_body_pct <= 0.3
        and gaps_down
        and c3_bullish
        and c3_body_pct >= 0.5
        and c3["close"] > c1_mid
    ):
        return {
            "detected": True,
            "name": "Morning Star",
            "emoji": "🌅",
            "signal": "BUY",
            "description": "Morning Star — tín hiệu đảo chiều tăng mạnh 3 nến. Seller kiệt sức, buyer tiếp quản.",
            "pattern_type": "morning_star",
        }
    return _empty_result()


def detect_evening_star(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return _empty_result()

    c1 = df.iloc[-3]
    c2 = df.iloc[-2]
    c3 = df.iloc[-1]

    _, _, _, _, c1_bullish, _, c1_body_pct = analyze_candle(c1)
    _, _, _, _, _, _, c2_body_pct = analyze_candle(c2)
    _, _, _, _, _, c3_bearish, c3_body_pct = analyze_candle(c3)

    c1_mid = (c1["open"] + c1["close"]) / 2
    gaps_up = min(c2["open"], c2["close"]) > c1["close"]

    if (
        c1_bullish
        and c1_body_pct >= 0.5
        and c2_body_pct <= 0.3
        and gaps_up
        and c3_bearish
        and c3_body_pct >= 0.5
        and c3["close"] < c1_mid
    ):
        return {
            "detected": True,
            "name": "Evening Star",
            "emoji": "🌆",
            "signal": "SELL",
            "description": "Evening Star — tín hiệu đảo chiều giảm mạnh 3 nến. Buyer kiệt sức, seller tiếp quản.",
            "pattern_type": "evening_star",
        }
    return _empty_result()


def detect_all_patterns(df: pd.DataFrame) -> list[dict]:
    """Run all pattern detections on df. Returns detected patterns only."""
    if df.empty:
        return []

    detectors = [
        detect_hammer,
        detect_shooting_star,
        detect_doji,
        detect_bullish_engulfing,
        detect_bearish_engulfing,
        detect_pin_bar,
        detect_inside_bar,
        detect_morning_star,
        detect_evening_star,
    ]

    results = []
    for detector in detectors:
        result = detector(df)
        if result["detected"]:
            results.append(result)
    return results
