import pandas as pd


def find_swings(df: pd.DataFrame, window: int = 5) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Find swing highs and lows using a rolling window.
    A swing high: candle whose high is highest in window on both sides
    A swing low: candle whose low is lowest in window on both sides

    Returns two lists of (index, price) tuples — last 10 swings of each type.
    """
    if len(df) < window * 2 + 1:
        return [], []

    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []

    highs = df["high"].values
    lows = df["low"].values

    for i in range(window, len(df) - window):
        left_high = highs[i - window : i]
        right_high = highs[i + 1 : i + window + 1]
        if highs[i] >= left_high.max() and highs[i] >= right_high.max():
            swing_highs.append((i, float(highs[i])))

        left_low = lows[i - window : i]
        right_low = lows[i + 1 : i + window + 1]
        if lows[i] <= left_low.min() and lows[i] <= right_low.min():
            swing_lows.append((i, float(lows[i])))

    return swing_highs[-10:], swing_lows[-10:]


def detect_trend(
    swing_highs: list[tuple[int, float]], swing_lows: list[tuple[int, float]]
) -> dict:
    """
    Analyze last 2 swing highs and 2 swing lows to determine trend.

    UPTREND:   Higher High + Higher Low
    DOWNTREND: Lower High  + Lower Low
    SIDEWAYS:  Mixed or insufficient data
    """
    hh = hl = lh = ll = False

    if len(swing_highs) >= 2:
        prev_high = swing_highs[-2][1]
        last_high = swing_highs[-1][1]
        hh = last_high > prev_high
        lh = last_high < prev_high

    if len(swing_lows) >= 2:
        prev_low = swing_lows[-2][1]
        last_low = swing_lows[-1][1]
        hl = last_low > prev_low
        ll = last_low < prev_low

    if hh and hl:
        return {
            "trend": "UPTREND",
            "emoji": "📈",
            "description": "Xu hướng tăng — đỉnh cao hơn (HH) và đáy cao hơn (HL). Buyer đang kiểm soát.",
            "hh": hh,
            "hl": hl,
            "lh": lh,
            "ll": ll,
        }

    if lh and ll:
        return {
            "trend": "DOWNTREND",
            "emoji": "📉",
            "description": "Xu hướng giảm — đỉnh thấp hơn (LH) và đáy thấp hơn (LL). Seller đang kiểm soát.",
            "hh": hh,
            "hl": hl,
            "lh": lh,
            "ll": ll,
        }

    return {
        "trend": "SIDEWAYS",
        "emoji": "➡️",
        "description": "Sideways — cấu trúc swing không rõ ràng hoặc thiếu dữ liệu. Thị trường đang tích lũy hoặc đảo chiều.",
        "hh": hh,
        "hl": hl,
        "lh": lh,
        "ll": ll,
    }


def _cluster_prices(prices: list[float], tolerance: float) -> list[tuple[float, int]]:
    if not prices:
        return []

    sorted_prices = sorted(prices)
    clusters: list[list[float]] = [[sorted_prices[0]]]

    for price in sorted_prices[1:]:
        cluster_avg = sum(clusters[-1]) / len(clusters[-1])
        if abs(price - cluster_avg) / cluster_avg <= tolerance:
            clusters[-1].append(price)
        else:
            clusters.append([price])

    return [(sum(cluster) / len(cluster), len(cluster)) for cluster in clusters]


def find_sr_zones(
    df: pd.DataFrame,
    swing_highs: list[tuple[int, float]],
    swing_lows: list[tuple[int, float]],
    tolerance: float = 0.003,
) -> list[dict]:
    """
    Cluster swing points into S/R zones.

    Returns up to 5 nearest zones (mix of support below and resistance above),
    sorted by distance from current price.
    """
    all_zones = _build_sr_zones(df, swing_highs, swing_lows, tolerance)
    if not all_zones:
        return []

    current_price = float(df.iloc[-1]["close"])
    supports = sorted(
        [z for z in all_zones if z["type"] == "SUPPORT"],
        key=lambda z: z["distance_pct"],
    )
    resistances = sorted(
        [z for z in all_zones if z["type"] == "RESISTANCE"],
        key=lambda z: z["distance_pct"],
    )

    selected: list[dict] = []
    for group in (supports[:3], resistances[:3]):
        for zone in group:
            if zone not in selected:
                selected.append(zone)
            if len(selected) >= 5:
                break
        if len(selected) >= 5:
            break

    selected.sort(key=lambda z: z["distance_pct"])
    return selected[:5]


def _build_sr_zones(
    df: pd.DataFrame,
    swing_highs: list[tuple[int, float]],
    swing_lows: list[tuple[int, float]],
    tolerance: float,
) -> list[dict]:
    if df.empty:
        return []

    current_price = float(df.iloc[-1]["close"])
    all_prices = [price for _, price in swing_highs] + [price for _, price in swing_lows]
    if not all_prices:
        return []

    zones: list[dict] = []
    for zone_price, strength in _cluster_prices(all_prices, tolerance):
        zone_type = "RESISTANCE" if zone_price > current_price else "SUPPORT"
        distance_pct = abs(zone_price - current_price) / current_price * 100
        zones.append(
            {
                "price": zone_price,
                "type": zone_type,
                "strength": strength,
                "distance_pct": distance_pct,
            }
        )
    return zones


def _determine_price_position(
    current_price: float,
    nearest_support: float | None,
    nearest_resistance: float | None,
    sr_zones: list[dict],
) -> str:
    if nearest_support is not None:
        support_dist = abs(current_price - nearest_support) / current_price
        if support_dist <= 0.01:
            return "AT_SUPPORT"

    if nearest_resistance is not None:
        resistance_dist = abs(current_price - nearest_resistance) / current_price
        if resistance_dist <= 0.01:
            return "AT_RESISTANCE"

    resistances = [z for z in sr_zones if z["type"] == "RESISTANCE"]
    supports = [z for z in sr_zones if z["type"] == "SUPPORT"]

    if resistances and all(current_price > z["price"] for z in resistances):
        return "ABOVE_ALL"
    if supports and all(current_price < z["price"] for z in supports):
        return "BELOW_ALL"

    return "MIDDLE"


def get_price_action_context(df: pd.DataFrame) -> dict:
    """Full price action analysis combining swings, trend, and S/R zones."""
    if df.empty:
        return {
            "trend": detect_trend([], []),
            "swing_highs": [],
            "swing_lows": [],
            "sr_zones": [],
            "nearest_support": None,
            "nearest_resistance": None,
            "current_price": 0.0,
            "price_position": "MIDDLE",
        }

    swing_highs, swing_lows = find_swings(df)
    trend = detect_trend(swing_highs, swing_lows)
    all_zones = _build_sr_zones(df, swing_highs, swing_lows, tolerance=0.003)
    sr_zones = find_sr_zones(df, swing_highs, swing_lows)
    current_price = float(df.iloc[-1]["close"])

    supports = [z for z in all_zones if z["type"] == "SUPPORT"]
    resistances = [z for z in all_zones if z["type"] == "RESISTANCE"]

    nearest_support = min(supports, key=lambda z: z["distance_pct"])["price"] if supports else None
    nearest_resistance = (
        min(resistances, key=lambda z: z["distance_pct"])["price"] if resistances else None
    )

    price_position = _determine_price_position(
        current_price, nearest_support, nearest_resistance, all_zones
    )

    return {
        "trend": trend,
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
        "sr_zones": sr_zones,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "current_price": current_price,
        "price_position": price_position,
    }
