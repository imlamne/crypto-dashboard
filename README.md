# 📊 Crypto Market Intelligence Dashboard

A personal crypto analytics platform built for learning technical analysis
and supporting trading decisions. Features automated data pipeline from
Binance API, technical indicators, price action analysis, candlestick pattern
detection, support/resistance zones, and strategy backtesting.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Binance](https://img.shields.io/badge/Data-Binance%20API-yellow)

## ✨ Features
- **Automated Data Pipeline** — fetches OHLCV from Binance API + Yahoo Finance (gold), stores in SQLite, auto-updates every 15 minutes
- **Technical Indicators** — MA (7/25/99), RSI (14), Bollinger Bands (20)
- **Price Action Analysis** — 9 candlestick patterns, swing detection, S/R zones
- **Multi-Timeframe Signals** — 15m / 1h / 4h / 1d confluence analysis
- **Action Suggestion** — entry, SL, TP levels based on Price Action + Indicators
- **Strategy Backtesting** — equity curve, win rate, drawdown analysis
- **Interactive Dashboard** — Streamlit + Plotly

## 📈 Supported Assets
| Symbol | Asset | Source |
|--------|-------|--------|
| BTCUSDT | Bitcoin | Binance API |
| ETHUSDT | Ethereum | Binance API |
| BNBUSDT | BNB | Binance API |
| SOLUSDT | Solana | Binance API |
| XAUUSD | Gold (Futures) | Yahoo Finance |

## 🛠 Tech Stack
| Layer | Technology |
|-------|-----------|
| Data | Python, requests, yfinance |
| Storage | SQLite, SQLAlchemy |
| Analysis | pandas, ta |
| Scheduling | APScheduler |
| Dashboard | Streamlit, Plotly |

## 📁 Project Structure
```
crypto-dashboard/
├── src/
│   ├── fetcher.py       # Binance API + Yahoo Finance
│   ├── database.py      # SQLite with upsert logic
│   ├── analysis.py      # MA, RSI, Bollinger Bands
│   ├── signals.py       # Multi-timeframe signal engine
│   ├── patterns.py      # 9 candlestick patterns
│   ├── price_action.py  # Swing detection, S/R zones
│   ├── backtest.py      # Strategy backtesting engine
│   └── dashboard.py     # Streamlit UI
├── config.py
├── main.py              # Pipeline + scheduler
├── start.bat            # One-click launcher (Windows)
└── requirements.txt
```

## 🚀 Quick Start

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run — Windows (one click):**
Double-click `start.bat`

**Or manually:**
```bash
# Terminal 1 — data pipeline + scheduler
python main.py

# Terminal 2 — dashboard
streamlit run src/dashboard.py
```

Open browser: http://localhost:8501

## 💡 Key Insights from Backtesting

- RSI+BB strategy underperforms on 15m (avg hold: 1-2 candles, BB width too narrow)
- Better risk-adjusted returns on 4h/1d timeframes
- Multi-timeframe confluence significantly reduces false signals
- Price action confirmation at S/R zones improves entry quality

## ⚠️ Disclaimer
Educational purposes only. Not financial advice.
Always manage risk — never trade more than you can afford to lose.

Built by Thanh Lam (Lenna) Nguyen
