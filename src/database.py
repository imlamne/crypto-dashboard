from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import DeclarativeBase, Session

import config


class Base(DeclarativeBase):
    pass


class OHLCV(Base):
    __tablename__ = "ohlcv"
    __table_args__ = (
        UniqueConstraint(
            "symbol", "interval", "open_time", name="uix_ohlcv_symbol_interval_open_time"
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    interval = Column(String, nullable=False)
    open_time = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    close_time = Column(DateTime, nullable=False)


engine = create_engine(f"sqlite:///{config.DATABASE_PATH}")


def init_db() -> None:
    Path(config.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def save_candles(df: pd.DataFrame, symbol: str, interval: str) -> None:
    if df.empty:
        return

    records = df.copy()
    records["open_time"] = pd.to_datetime(records["open_time"])
    records["close_time"] = pd.to_datetime(records["close_time"])
    records["symbol"] = symbol
    records["interval"] = interval

    columns = [
        "symbol",
        "interval",
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
    ]
    payload = records[columns].to_dict(orient="records")

    stmt = insert(OHLCV).values(payload)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "interval", "open_time"],
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
            "close_time": stmt.excluded.close_time,
        },
    )

    with engine.begin() as conn:
        conn.execute(stmt)


def get_candles(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    with Session(engine) as session:
        rows = session.scalars(
            select(OHLCV)
            .where(OHLCV.symbol == symbol, OHLCV.interval == interval)
            .order_by(OHLCV.open_time.desc())
            .limit(limit)
        ).all()

    if not rows:
        return pd.DataFrame(
            columns=["open_time", "open", "high", "low", "close", "volume", "close_time"]
        )

    data = [
        {
            "open_time": row.open_time,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
            "close_time": row.close_time,
        }
        for row in reversed(rows)
    ]
    return pd.DataFrame(data)


def get_available_symbols() -> list[str]:
    """Return all distinct symbols that have data in the DB."""
    with Session(engine) as session:
        rows = session.execute(
            select(OHLCV.symbol).distinct().order_by(OHLCV.symbol)
        ).all()
    return [row[0] for row in rows]
