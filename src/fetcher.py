import pandas as pd
import requests
import yfinance as yf

import config

COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time"]

YFINANCE_INTERVAL_MAP = {
    "15m": ("15m", "60d"),
    "1h": ("1h", "730d"),
    "4h": ("1h", "730d"),
    "1d": ("1d", "max"),
}

INTERVAL_DURATIONS = {
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}

def _empty_candles() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def fetch_candles(symbol: str, interval: str) -> pd.DataFrame:
    empty = _empty_candles()

    url = f"{config.BINANCE_BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": config.DEFAULT_CANDLE_LIMIT,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        print(f"Error fetching candles for {symbol} ({interval}): {exc}")
        return empty

    if not data:
        return empty

    df = pd.DataFrame(
        {
            "open_time": [row[0] for row in data],
            "open": [row[1] for row in data],
            "high": [row[2] for row in data],
            "low": [row[3] for row in data],
            "close": [row[4] for row in data],
            "volume": [row[5] for row in data],
            "close_time": [row[6] for row in data],
        }
    )

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)

    return df


def fetch_gold_candles(interval: str) -> pd.DataFrame:
    empty = _empty_candles()
    duration = INTERVAL_DURATIONS[interval]

    try:
        yf_interval, period = YFINANCE_INTERVAL_MAP[interval]
        df = yf.download(
            "GC=F",
            interval=yf_interval,
            period=period,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            return empty

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        if interval == "4h":
            df = df.resample("4h").agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            ).dropna()

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        df = df.reset_index()
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "open_time"})
        df["open_time"] = pd.to_datetime(df["open_time"])
        if df["open_time"].dt.tz is not None:
            df["open_time"] = df["open_time"].dt.tz_localize(None)

        df["close_time"] = df["open_time"] + duration
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)

        df = df.sort_values("open_time").reset_index(drop=True)
        df = df[COLUMNS].tail(config.DEFAULT_CANDLE_LIMIT).reset_index(drop=True)
        print(f"  Gold ({interval}): fetched {len(df)} rows")
        return df

    except Exception as exc:
        print(f"Error fetching gold candles ({interval}): {exc}")
        return empty