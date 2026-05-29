import config
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.database import init_db, save_candles
from src.fetcher import fetch_candles, fetch_gold_candles


def run() -> None:
    init_db()

    for symbol in config.TRADING_PAIRS:
        if symbol == "XAUUSD":
            continue  # handled separately
        for interval in config.TIMEFRAMES:
            print(f"Fetching {symbol} {interval}...")
            df = fetch_candles(symbol, interval)

            if df.empty:
                print(f"Warning: no data returned for {symbol} {interval}, skipping.")
                continue

            save_candles(df, symbol, interval)
            print(f"Saved {len(df)} rows for {symbol} {interval}.")

    # Fetch gold separately via Alpha Vantage
    print("Fetching gold (XAUUSD)...")
    for interval in config.TIMEFRAMES:
        df = fetch_gold_candles(interval)
        if df.empty:
            print(f"  Warning: no data for XAUUSD {interval}, skipping.")
            continue
        save_candles(df, "XAUUSD", interval)
        print(f"  Saved {len(df)} rows for XAUUSD {interval}.")

    print("Pipeline complete.")


def start_scheduler() -> None:
    scheduler = BlockingScheduler()
    scheduler.add_job(run, IntervalTrigger(minutes=config.SCHEDULER_INTERVAL_MINUTES))
    print(
        f"Scheduler started. Fetching every {config.SCHEDULER_INTERVAL_MINUTES} minutes. "
        "Press Ctrl+C to stop."
    )
    scheduler.start()


if __name__ == "__main__":
    run()
    start_scheduler()
