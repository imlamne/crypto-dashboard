import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide", page_title="Crypto Dashboard")

from src.analysis import add_all_indicators
from src.backtest import get_equity_curve, run_backtest
from src.database import get_available_symbols, get_candles
from src.patterns import detect_all_patterns
from src.price_action import get_price_action_context
from src.signals import get_all_signals, get_mtf_conclusion, get_signal


@st.cache_data(ttl=300)
def load_signals():
    return get_all_signals()


@st.cache_data(ttl=300)
def load_candles(symbol, interval, limit):
    return get_candles(symbol, interval, limit)


@st.cache_data(ttl=300)
def load_backtest(symbol, interval):
    df = load_candles(symbol, interval, 500)
    if df.empty:
        return None
    df = add_all_indicators(df)
    return run_backtest(df, symbol, interval)


BACKTEST_INITIAL_CAPITAL = 1000.0

RESULT_LABELS = {
    "WIN": "WIN 🟢",
    "LOSS": "LOSS 🔴",
    "BREAKEVEN": "BREAKEVEN ⚪",
}

MTF_EMOJI = {
    "STRONG BUY": "🟢",
    "WEAK BUY": "🟡",
    "STRONG SELL": "🔴",
    "WEAK SELL": "🟠",
    "NEUTRAL": "⚪",
    "CONFLICTED": "⚠️",
}

MTF_COLORS = {
    "STRONG BUY": "#3fb950",
    "WEAK BUY": "#56d364",
    "STRONG SELL": "#f85149",
    "WEAK SELL": "#ff7b72",
    "CONFLICTED": "#d29922",
    "NEUTRAL": "#8b949e",
}

# ---------------------------------------------------------------------------
# Helpers — price action
# ---------------------------------------------------------------------------
PRICE_POSITION_TEXT = {
    "AT_SUPPORT": "Giá đang test vùng **hỗ trợ** — cơ hội mua nếu có xác nhận từ mẫu nến.",
    "AT_RESISTANCE": "Giá đang test vùng **kháng cự** — thận trọng khi mua, chờ breakout hoặc pullback.",
    "MIDDLE": "Giá đang ở **giữa** vùng S/R — chưa có setup rõ ràng.",
    "ABOVE_ALL": "Giá **breakout** trên mọi kháng cự — momentum tăng mạnh.",
    "BELOW_ALL": "Giá **breakdown** dưới mọi hỗ trợ — momentum giảm mạnh.",
}

PATTERN_BOX_STYLE = {
    "BUY": ("#dafbe1", "#1a4731", "#3fb950"),
    "SELL": ("#ffebe9", "#4a1c1c", "#f85149"),
    "NEUTRAL": ("#f6f8fa", "#57606a", "#8b949e"),
}


def _format_num(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _calc_rr(entry: float, sl: float, tp: float) -> float:
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    return reward / risk if risk > 0 else 0.0


def _zone_at_price(price: float | None, zones: list[dict]) -> dict | None:
    if price is None:
        return None
    for zone in zones:
        if abs(zone["price"] - price) / price <= 0.005:
            return zone
    return None


def _strength_stars(strength: int) -> str:
    return "★" * max(1, strength)


def _next_resistance(current: float, resistance: float | None, zones: list[dict]) -> float | None:
    above = sorted(
        [z["price"] for z in zones if z["type"] == "RESISTANCE" and z["price"] > current + 1e-9],
        key=lambda p: p,
    )
    if len(above) >= 2:
        return above[1]
    if above:
        return above[0] * 1.02
    if resistance:
        return resistance * 1.02
    return current * 1.03


def get_pa_action(
    pa_context: dict, patterns: list[dict], signal_dict: dict
) -> tuple[str, dict]:
    trend = pa_context["trend"]["trend"]
    position = pa_context["price_position"]
    buy_patterns = [p for p in patterns if p["signal"] == "BUY"]
    sell_patterns = [p for p in patterns if p["signal"] == "SELL"]
    indicator_score = signal_dict["score"]

    meta = {
        "trend": trend,
        "position": position,
        "current": pa_context["current_price"],
        "support": pa_context["nearest_support"],
        "resistance": pa_context["nearest_resistance"],
        "buy_patterns": buy_patterns,
        "sell_patterns": sell_patterns,
        "signal_dict": signal_dict,
        "sr_zones": pa_context["sr_zones"],
        "trend_info": pa_context["trend"],
    }

    if (
        trend == "UPTREND"
        and position == "AT_SUPPORT"
        and buy_patterns
        and indicator_score >= 2
    ):
        return "strong_buy", meta

    if (
        (position == "AT_RESISTANCE" and sell_patterns and indicator_score <= -2)
        or (trend == "DOWNTREND" and position == "AT_RESISTANCE")
    ):
        return "strong_sell", meta

    if trend == "UPTREND" and position == "AT_RESISTANCE":
        return "caution_resistance", meta

    if trend == "UPTREND" and position == "AT_SUPPORT" and not buy_patterns:
        return "wait_buy", meta

    if position == "AT_SUPPORT" and signal_dict["label"] in ("WEAK BUY", "NEUTRAL"):
        return "wait_buy", meta

    return "neutral", meta


def _render_pattern_box(pattern: dict) -> None:
    bg, border, text = PATTERN_BOX_STYLE.get(pattern["signal"], PATTERN_BOX_STYLE["NEUTRAL"])
    st.markdown(
        f"""
<div style="background:{bg};border:1px solid {border};border-radius:8px;
padding:12px 14px;margin-bottom:10px;color:{text};">
  <div style="font-size:15px;font-weight:700;margin-bottom:4px;">
    {pattern["emoji"]} {pattern["name"]}
  </div>
  <div style="font-size:13px;color:#24292f;margin-bottom:6px;">{pattern["description"]}</div>
  <div style="font-size:12px;font-weight:600;">Signal: {pattern["signal"]}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def build_sr_chart(df: pd.DataFrame, sr_zones: list[dict], current_price: float):
    chart_df = df.tail(100).copy()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df["open_time"],
            y=chart_df["close"],
            mode="lines",
            line=dict(color="#0969da", width=2),
            name="Price",
        )
    )

    for zone in sr_zones:
        is_support = zone["type"] == "SUPPORT"
        color = "#3fb950" if is_support else "#f85149"
        prefix = "S" if is_support else "R"
        stars = _strength_stars(zone["strength"])
        fig.add_hline(
            y=zone["price"],
            line=dict(color=color, dash="dash", width=zone["strength"]),
            annotation_text=f"{prefix}: {zone['price']:,.2f} {stars}",
            annotation_font=dict(color=color, size=11),
            annotation_position="right",
        )

    fig.add_hline(
        y=current_price,
        line=dict(color="#57606a", width=1, dash="dot"),
        annotation_text=f"Now: {current_price:,.2f}",
        annotation_font=dict(color="#57606a", size=11),
        annotation_position="left",
    )

    _apply_chart_layout(
        fig,
        title="Support & Resistance Zones",
        height=400,
        legend_extra=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _render_pa_action(action_type: str, meta: dict) -> None:
    current = meta["current"]
    support = meta["support"]
    resistance = meta["resistance"]
    zones = meta["sr_zones"]
    signal_dict = meta["signal_dict"]
    trend_info = meta["trend_info"]

    if action_type == "caution_resistance" and resistance:
        res_zone = _zone_at_price(resistance, zones) or {"strength": 1}
        strength = res_zone["strength"]
        next_res = _next_resistance(current, resistance, zones)
        entry_break = resistance * 1.005
        sl_break = resistance * 0.990
        rr1 = _calc_rr(entry_break, sl_break, next_res or resistance * 1.03)

        entry_pull = (support or current) * 1.005 if support else current
        sl_pull = (support or current) * 0.985 if support else current * 0.985
        rr2 = _calc_rr(entry_pull, sl_pull, resistance) if support else 0.0

        st.markdown(
            f"""
### ⚠️ THẬN TRỌNG — Giá đang tại vùng kháng cự

**📊 Phân tích:**
- Trend: **{meta['trend']}** {trend_info['emoji']} (tích cực)
- Nhưng giá đang sát Resistance **{resistance:,.2f}** (đã test **{strength}** lần)
- Resistance càng nhiều lần test → càng khó phá vỡ

**🎯 Kịch bản 1 — Breakout (giá phá kháng cự):**
- Điều kiện: nến đóng cửa **TRÊN {resistance:,.2f}** với volume lớn
- → Entry: **{entry_break:,.2f}** (sau khi xác nhận breakout)
- → SL: **{sl_break:,.2f}** (dưới vùng kháng cự cũ)
- → TP1: **{(next_res or resistance * 1.03):,.2f}**
- → R:R: **{rr1:.1f}**

**🎯 Kịch bản 2 — Pullback to Support (giá quay về hỗ trợ):**
- Điều kiện: giá quay về vùng **{(support or 0):,.2f}**
  + xuất hiện mẫu nến đảo chiều (Hammer, Engulfing...)
- → Entry: **{entry_pull:,.2f}**
- → SL: **{sl_pull:,.2f}**
- → TP1: **{resistance:,.2f}** (kháng cự cũ)
- → R:R: **{rr2:.1f}**

**⏳ Hiện tại: CHƯA VÀO LỆNH**  
Chờ một trong 2 kịch bản xác nhận
"""
        )
        return

    if action_type == "strong_buy" and support and meta["buy_patterns"]:
        sup_zone = _zone_at_price(support, zones) or {"strength": 1}
        pattern = meta["buy_patterns"][0]
        entry = support * 1.005
        sl = support * 0.985
        sl_pct = abs(entry - sl) / entry * 100
        tp1 = resistance or current * 1.03
        tp2 = _next_resistance(current, resistance, zones) or tp1 * 1.02
        rr1 = _calc_rr(entry, sl, tp1)
        rr2 = _calc_rr(entry, sl, tp2)

        st.markdown(
            f"""
### 🟢 STRONG BUY — Hội tụ đủ điều kiện Price Action

**✅ Xác nhận:**
- Trend: **UPTREND** 📈
- Giá tại vùng Support **{support:,.2f}** (strength: {sup_zone['strength']})
- Mẫu nến: **{pattern['name']}** — {pattern['description']}
- Indicator: **{signal_dict['label']}** ({signal_dict['score']}/4 tín hiệu)

**📍 Entry:** ~**{entry:,.2f}**  
**🛡️ SL:** ~**{sl:,.2f}** ({sl_pct:.1f}% dưới support)  
**🎯 TP1:** ~**{tp1:,.2f}** (kháng cự gần nhất — R:R {rr1:.1f})  
**🎯 TP2:** ~**{tp2:,.2f}** (kháng cự tiếp theo — R:R {rr2:.1f})
"""
        )
        return

    if action_type == "strong_sell" and resistance:
        res_zone = _zone_at_price(resistance, zones) or {"strength": 1}
        pattern = meta["sell_patterns"][0] if meta["sell_patterns"] else None
        entry = resistance * 0.995
        sl = resistance * 1.015
        tp1 = support or current * 0.97
        rr1 = _calc_rr(entry, sl, tp1)

        pattern_line = (
            f"- Mẫu nến: **{pattern['name']}** — {pattern['description']}\n"
            if pattern
            else ""
        )
        st.markdown(
            f"""
### 🔴 STRONG SELL / TRÁNH MUA — Tín hiệu giảm mạnh

**📊 Phân tích:**
- Trend: **{meta['trend']}** {trend_info['emoji']}
- Giá tại vùng Resistance **{resistance:,.2f}** (strength: {res_zone['strength']})
{pattern_line}- Indicator: **{signal_dict['label']}** ({signal_dict['score']}/4 tín hiệu)

**📍 Entry (short):** ~**{entry:,.2f}**  
**🛡️ SL:** ~**{sl:,.2f}**  
**🎯 TP1:** ~**{tp1:,.2f}** — R:R **{rr1:.1f}**

**⏳ Khuyến nghị:** Không mua tại vùng này — chờ pullback hoặc xác nhận đảo chiều.
"""
        )
        return

    if action_type == "wait_buy" and support:
        st.markdown(
            f"""
### 🟡 CHỜ XÁC NHẬN — Setup gần hoàn chỉnh

**📊 Phân tích:**
- Trend: **{meta['trend']}** {trend_info['emoji']}
- Giá tại Support **{support:,.2f}** — vùng tiềm năng để mua
- Chưa có mẫu nến xác nhận (Hammer, Engulfing, Pin Bar...)
- Indicator: **{signal_dict['label']}** ({signal_dict['score']}/4 tín hiệu)

**👀 Chờ thêm:**
- Mẫu nến đảo chiều tăng xuất hiện tại support
- Volume xác nhận trên nến tăng
- Indicator score >= 2

**⏳ Hiện tại: CHƯA VÀO LỆNH**
"""
        )
        return

    st.markdown(
        f"""
### ⚪ QUAN SÁT — Chưa có tín hiệu rõ ràng

**📊 Giá đang ở giữa vùng S/R, chưa có setup rõ ràng**

**👀 Theo dõi:**
- Giá có về test Support **{(support or 0):,.2f}** không?
- Giá có breakout Resistance **{(resistance or 0):,.2f}** không?
- Mẫu nến xác nhận xuất hiện chưa?

- Trend hiện tại: **{meta['trend']}** {trend_info['emoji']}
- Indicator: **{signal_dict['label']}** ({signal_dict['score']}/4 tín hiệu)
"""
    )


def render_price_action_analysis(
    df: pd.DataFrame, symbol: str, interval: str
) -> None:
    pa_context = get_price_action_context(df)
    patterns = detect_all_patterns(df)
    signal_dict = get_signal(df)

    trend = pa_context["trend"]
    current = pa_context["current_price"]
    support = pa_context["nearest_support"]
    resistance = pa_context["nearest_resistance"]
    position = pa_context["price_position"]
    sr_zones = pa_context["sr_zones"]

    dist_support = abs(current - support) / current * 100 if support else None
    dist_resistance = abs(resistance - current) / current * 100 if resistance else None

    with st.expander(f"📊 Price Action Analysis — {symbol} {interval}", expanded=True):
        tab1, tab2, tab3 = st.tabs(
            ["🕯️ Pattern & Trend", "📍 Support & Resistance", "💡 Action Suggestion"]
        )

        with tab1:
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown("### Xu hướng (Trend)")
                st.markdown(
                    f"{trend['emoji']} **{trend['trend']}**\n\n{trend['description']}"
                )

                st.markdown("### Vị trí giá")
                st.markdown(PRICE_POSITION_TEXT.get(position, position))

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Current Price", f"{current:,.2f}")
                with m2:
                    st.metric(
                        "Nearest Support",
                        f"{support:,.2f}" if support else "N/A",
                        delta=f"-{dist_support:.2f}%" if dist_support is not None else None,
                        delta_color="off",
                    )
                with m3:
                    st.metric(
                        "Nearest Resistance",
                        f"{resistance:,.2f}" if resistance else "N/A",
                        delta=f"+{dist_resistance:.2f}%" if dist_resistance is not None else None,
                        delta_color="off",
                    )

                if support and dist_support is not None:
                    st.caption(f"Nearest Support: **{support:,.2f}** ({dist_support:.2f}% bên dưới)")
                if resistance and dist_resistance is not None:
                    st.caption(
                        f"Nearest Resistance: **{resistance:,.2f}** ({dist_resistance:.2f}% bên trên)"
                    )

            with col_right:
                st.markdown("### Mẫu nến hiện tại")
                if patterns:
                    for pattern in patterns:
                        _render_pattern_box(pattern)
                else:
                    st.info(
                        "Không có mẫu nến đặc biệt — thị trường đang trong trạng thái bình thường"
                    )

        with tab2:
            st.plotly_chart(
                build_sr_chart(df, sr_zones, current),
                use_container_width=True,
            )

            if sr_zones:
                rows = []
                for zone in sr_zones:
                    icon = "🟢" if zone["type"] == "SUPPORT" else "🔴"
                    label = f"{icon} {zone['type'].title()}"
                    sign = "+" if zone["type"] == "RESISTANCE" else "-"
                    rows.append(
                        {
                            "Zone": label,
                            "Price": f"{zone['price']:,.2f}",
                            "Type": zone["type"],
                            "Strength": _strength_stars(zone["strength"]),
                            "Distance": f"{sign}{zone['distance_pct']:.2f}%",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Chưa phát hiện vùng S/R rõ ràng từ swing points.")

        with tab3:
            action_type, meta = get_pa_action(pa_context, patterns, signal_dict)
            _render_pa_action(action_type, meta)
            st.markdown(
                """
---
⚠️ **Phân tích kỹ thuật — không phải lời khuyên tài chính.**  
Luôn đặt SL và chỉ rủi ro vốn bạn sẵn sàng mất.
"""
            )


def _backtest_metric(col, label: str, value: str, color: str) -> None:
    with col:
        st.markdown(
            f'<p style="color:#57606a;font-size:13px;margin:0 0 4px 0;">{label}</p>'
            f'<p style="color:{color};font-size:26px;font-weight:700;margin:0;">{value}</p>',
            unsafe_allow_html=True,
        )


def build_equity_curve_chart(trades: list[dict], initial_capital: float = BACKTEST_INITIAL_CAPITAL):
    curve = get_equity_curve(trades, initial_capital)
    trade_nums = [point[0] for point in curve]
    equities = [point[1] for point in curve]
    line_color = "#3fb950" if equities[-1] >= initial_capital else "#f85149"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trade_nums,
            y=equities,
            mode="lines+markers",
            line=dict(color=line_color, width=2),
            marker=dict(size=6, color=line_color),
            name="Equity",
        )
    )
    fig.add_hline(
        y=initial_capital,
        line=dict(color="#57606a", dash="dash", width=1),
        annotation_text=f"${initial_capital:,.0f}",
        annotation_font=dict(color="#57606a", size=11),
    )

    _apply_chart_layout(
        fig,
        title=f"Equity Curve — ${initial_capital:,.0f} starting capital",
        height=320,
        showlegend=False,
    )
    return fig


def _highlight_trade_row(row: pd.Series):
    result = str(row["Result"])
    if "WIN" in result:
        bg = "background-color: #dafbe1"
    elif "LOSS" in result:
        bg = "background-color: #ffebe9"
    else:
        bg = "background-color: #f6f8fa"
    return [bg] * len(row)


def render_backtesting(symbol: str, interval: str) -> None:
    with st.expander(f"📈 Backtesting — {symbol} {interval}", expanded=False):
        result = load_backtest(symbol, interval)
        if result is None:
            st.warning("Không đủ dữ liệu để chạy backtest. Chạy `python main.py` trước.")
            return

        trades = result["trades"]
        s = result["summary"]

        if s["total_trades"] == 0:
            st.info("Không có lệnh nào trong khoảng dữ liệu backtest.")
            st.info(
                """
📋 **Chiến lược đang test:**
- **BUY khi:** RSI < 40 + (MA7 cong lên HOẶC giá gần BB Lower)
- **EXIT khi:** Giá chạm BB Upper | RSI > 65 | MA7 đảo chiều | SL -2%

⚠️ Backtesting không đảm bảo kết quả tương lai. Past performance ≠ future results.
"""
            )
            return

        win_color = "#3fb950" if s["win_rate"] > 50 else "#f85149"
        return_color = "#3fb950" if s["total_return_pct"] >= 0 else "#f85149"

        m1, m2, m3, m4 = st.columns(4)
        _backtest_metric(m1, "Total Trades", str(s["total_trades"]), "#24292f")
        _backtest_metric(m2, "Win Rate", f"{s['win_rate']:.1f}%", win_color)
        _backtest_metric(m3, "Total Return", f"{s['total_return_pct']:+.2f}%", return_color)
        _backtest_metric(m4, "Max Drawdown", f"{s['max_drawdown_pct']:.2f}%", "#f85149")

        col_stats, col_chart = st.columns([1, 1])

        with col_stats:
            st.markdown("#### Trade stats")
            st.markdown(
                f"""
- **Avg Return per trade:** {s['avg_return_pct']:+.2f}%
- **Best Trade:** {s['best_trade_pct']:+.2f}%
- **Worst Trade:** {s['worst_trade_pct']:+.2f}%
- **Avg Hold (candles):** {s['avg_hold_candles']:.1f}
- **Winning / Losing:** {s['winning_trades']} / {s['losing_trades']}
"""
            )

        with col_chart:
            st.plotly_chart(
                build_equity_curve_chart(trades),
                use_container_width=True,
            )

        st.markdown("#### Trade log")
        log_rows = []
        for n, trade in enumerate(trades, start=1):
            log_rows.append(
                {
                    "#": n,
                    "Entry Time": trade["entry_time"],
                    "Entry Price": f"{trade['entry_price']:,.2f}",
                    "Exit Time": trade["exit_time"],
                    "Exit Price": f"{trade['exit_price']:,.2f}",
                    "Return %": f"{trade['return_pct']:+.2f}%",
                    "Result": RESULT_LABELS.get(trade["result"], trade["result"]),
                    "Exit Reason": trade["exit_reason"],
                }
            )
        log_df = pd.DataFrame(log_rows)
        st.dataframe(
            log_df.style.apply(_highlight_trade_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            """
📋 **Chiến lược đang test:**
- **BUY khi:** RSI < 40 + (MA7 cong lên HOẶC giá gần BB Lower)
- **EXIT khi:** Giá chạm BB Upper | RSI > 65 | MA7 đảo chiều | SL -2%

⚠️ Backtesting không đảm bảo kết quả tương lai.  
Past performance ≠ future results.
"""
        )


def _signal_cell_html(label: str, emoji: str) -> str:
    text = f"{emoji} {label}"
    if "BUY" in label:
        return f'<td style="background:#1a4731;color:#3fb950;font-weight:600;">{text}</td>'
    if "SELL" in label:
        return f'<td style="background:#4a1c1c;color:#f85149;font-weight:600;">{text}</td>'
    return f'<td style="background:transparent;color:#57606a;">{text}</td>'


def build_signal_table_html(signals: dict, symbols: list[str]) -> str:
    header_cells = "".join(f"<th>{tf}</th>" for tf in config.TIMEFRAMES)
    rows = []
    for sym in symbols:
        cells = [f"<td><strong>{sym}</strong></td>"]
        for tf in config.TIMEFRAMES:
            signal = signals.get((sym, tf), {"emoji": "⚪", "label": "NEUTRAL"})
            cells.append(_signal_cell_html(signal["label"], signal["emoji"]))
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
<table class="signal-table">
  <thead><tr><th>Symbol</th>{header_cells}</tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
"""


def render_mtf_conclusion(symbols: list[str]) -> None:
    st.markdown("### Multi-Timeframe Conclusion")
    if not symbols:
        st.info("No symbols in database.")
        return

    cols = st.columns(len(symbols))
    for col, sym in zip(cols, symbols):
        conclusion = get_mtf_conclusion(sym)
        emoji = MTF_EMOJI.get(conclusion, "⚪")
        color = MTF_COLORS.get(conclusion, "#8b949e")
        with col:
            st.markdown(
                f'<p style="font-size:12px;color:#57606a;margin-bottom:2px;">{sym}</p>'
                f'<p style="font-size:22px;font-weight:700;color:{color};margin:0;">{emoji} {conclusion}</p>',
                unsafe_allow_html=True,
            )


def _guide_metrics(df: pd.DataFrame) -> dict:
    row = df.iloc[-1]
    last_close = float(row["close"])
    last_ma7 = float(row["ma_7"]) if not pd.isna(row["ma_7"]) else last_close
    last_ma25 = float(row["ma_25"]) if not pd.isna(row["ma_25"]) else last_close
    last_ma99 = float(row["ma_99"]) if not pd.isna(row["ma_99"]) else last_close
    last_bb_upper = float(row["bb_upper"]) if not pd.isna(row["bb_upper"]) else last_close
    last_bb_middle = float(row["bb_middle"]) if not pd.isna(row["bb_middle"]) else last_close
    last_bb_lower = float(row["bb_lower"]) if not pd.isna(row["bb_lower"]) else last_close
    last_rsi = float(row["rsi"]) if not pd.isna(row["rsi"]) else 50.0

    bb_width_series = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"] * 100
    bb_width = (
        (last_bb_upper - last_bb_lower) / last_bb_middle * 100
        if last_bb_middle
        else 0.0
    )
    bb_width_avg = float(bb_width_series.dropna().mean()) if not bb_width_series.dropna().empty else bb_width

    vol_decreasing = False
    if len(df) >= 11:
        vol_decreasing = row["volume"] < df["volume"].iloc[-11:-1].mean()

    ma7_rising = False
    if len(df) >= 2:
        ma_diff = df["ma_7"].diff().iloc[-1]
        ma7_rising = not pd.isna(ma_diff) and ma_diff > 0

    bb_range = last_bb_upper - last_bb_lower
    pct_position = (
        (last_close - last_bb_lower) / bb_range * 100 if bb_range > 0 else 50.0
    )
    pct_position = max(0.0, min(100.0, pct_position))

    sl_buy = last_bb_lower * 0.985
    sl_pct = abs(last_close - sl_buy) / last_close * 100 if last_close else 0.0
    tp1_buy = last_bb_middle
    risk_amount = 10.0
    position_size = risk_amount / (last_close - sl_buy) if last_close > sl_buy else 0.0

    return {
        "last_close": last_close,
        "last_ma7": last_ma7,
        "last_ma25": last_ma25,
        "last_ma99": last_ma99,
        "last_bb_upper": last_bb_upper,
        "last_bb_middle": last_bb_middle,
        "last_bb_lower": last_bb_lower,
        "last_rsi": last_rsi,
        "bb_width": bb_width,
        "bb_width_avg": bb_width_avg,
        "vol_decreasing": vol_decreasing,
        "ma7_rising": ma7_rising,
        "pct_position": pct_position,
        "sl_buy": sl_buy,
        "sl_pct": sl_pct,
        "tp1_buy": tp1_buy,
        "position_size": position_size,
    }


def render_analysis_guide(df: pd.DataFrame, symbol: str, interval: str) -> None:
    m = _guide_metrics(df)

    with st.expander("📚 Hướng dẫn Phân tích", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Bước 1 — Đọc Trend",
                "📍 Bước 2 — Tìm vùng quan trọng",
                "🔍 Bước 3 — Xác nhận tín hiệu",
                "💰 Bước 4 — Quản lý lệnh",
            ]
        )

        with tab1:
            st.markdown(
                """
<div style="font-size:14px;">

## Đọc xu hướng tổng thể (Top-down)
Luôn bắt đầu từ khung lớn → nhỏ dần: **1d → 4h → 1h → 15m**

### Nhìn vào MA99 (đường đỏ):
✅ Giá **TRÊN MA99** + MA99 dốc lên → **Uptrend** — ưu tiên tìm lệnh BUY  
❌ Giá **DƯỚI MA99** + MA99 dốc xuống → **Downtrend** — ưu tiên tìm lệnh SELL  
⚪ Giá quanh MA99 → **Sideways** — thận trọng, chờ rõ hơn

### Nhìn vào Bollinger Bands:
📈 **BB giãn rộng** → Trend đang mạnh, momentum cao  
📉 **BB thu hẹp** → Sắp có bứt phá lớn, chờ tín hiệu  
➡️ **BB bình thường** → Thị trường đang bình ổn

### Nhìn vào MA7 vs MA25:
🟢 MA7 cắt lên trên MA25 (Golden Cross) → Tín hiệu BUY  
🔴 MA7 cắt xuống dưới MA25 (Death Cross) → Tín hiệu SELL

</div>
""",
                unsafe_allow_html=True,
            )
            st.info(
                f"""
📍 **{symbol} {interval}** hiện tại:
• Giá vs MA99: {'📈 Trên' if m['last_close'] > m['last_ma99'] else '📉 Dưới'} MA99 ({m['last_ma99']:,.2f})
• MA7 vs MA25: {'🟢 MA7 trên MA25' if m['last_ma7'] > m['last_ma25'] else '🔴 MA7 dưới MA25'}
• BB width: {m['bb_width']:.2f}% {'(đang giãn rộng 📈)' if m['bb_width'] > m['bb_width_avg'] else '(đang thu hẹp ⚠️)'}
"""
            )
            st.info(
                "💡 **Tip:** Khung **1d** cho bức tranh lớn nhất. "
                "Nếu 1d đang downtrend, đừng cố tìm lệnh BUY trên 15m."
            )

        with tab2:
            st.markdown(
                f"""
<div style="font-size:14px;">

## Xác định giá đang ở vùng nào

### Bollinger Bands — 3 vùng chính:

🔵 **Vùng BB Upper** ({m['last_bb_upper']:,.2f}):
→ Vùng kháng cự — giá thường quay đầu giảm  
→ RSI thường > 60-70 tại đây  
→ **Hành động:** cân nhắc chốt lời nếu đang giữ lệnh BUY

⚪ **Vùng BB Middle** ({m['last_bb_middle']:,.2f}):
→ Đường trung bình — vùng quyết định  
→ Giá trên BB Middle → xu hướng tăng ngắn hạn  
→ Giá dưới BB Middle → xu hướng giảm ngắn hạn

🔵 **Vùng BB Lower** ({m['last_bb_lower']:,.2f}):
→ Vùng hỗ trợ — giá thường nảy lên  
→ RSI thường < 30-40 tại đây  
→ **Hành động:** cân nhắc tìm lệnh BUY nếu có xác nhận

### Giá hiện tại đang ở đâu?

</div>
""",
                unsafe_allow_html=True,
            )
            st.progress(
                m["pct_position"] / 100,
                text=f"Giá đang ở {m['pct_position']:.0f}% trong dải BB (0%=Lower, 100%=Upper)",
            )

        with tab3:
            st.markdown(
                """
<div style="font-size:14px;">

## Checklist xác nhận trước khi vào lệnh

Không bao giờ vào lệnh chỉ dựa vào 1 indicator.  
Cần **ít nhất 3/4** yếu tố đồng thuận:

### Checklist BUY:
☐ RSI < 35 (oversold, có thể < 40 nếu uptrend mạnh)  
☐ Giá ở hoặc dưới BB Lower  
☐ Volume đang **GIẢM** (áp lực bán hạ) **HOẶC** Volume đột biến **TĂNG** trên nến xanh  
☐ MA7 bắt đầu cong lên (momentum đổi chiều)  
☐ **[BONUS]** Khung lớn hơn cũng đang BUY/NEUTRAL

### Checklist SELL:
☐ RSI > 65 (overbought)  
☐ Giá ở hoặc trên BB Upper  
☐ Volume đang **TĂNG** (áp lực bán mạnh)  
☐ MA7 bắt đầu cong xuống  
☐ **[BONUS]** Khung lớn hơn cũng đang SELL/NEUTRAL

</div>
""",
                unsafe_allow_html=True,
            )

            checks = {
                f"RSI = {m['last_rsi']:.1f}": m["last_rsi"] < 35,
                f"Giá vs BB Lower: {m['last_close']:,.0f} vs {m['last_bb_lower']:,.0f}": m[
                    "last_close"
                ]
                <= m["last_bb_lower"] * 1.01,
                "Volume giảm": m["vol_decreasing"],
                "MA7 slope tăng": m["ma7_rising"],
            }
            passed_count = sum(checks.values())
            st.markdown('<div style="font-size:14px;">', unsafe_allow_html=True)
            for label, passed in checks.items():
                st.markdown(f"{'🟢' if passed else '🔴'} {label}")
            st.markdown("</div>", unsafe_allow_html=True)

            if passed_count >= 3:
                st.success(f"✅ {passed_count}/4 yếu tố BUY đồng thuận — tín hiệu đủ mạnh để cân nhắc.")
            elif passed_count >= 2:
                st.info(f"ℹ️ {passed_count}/4 yếu tố — cần thêm xác nhận trước khi vào lệnh.")
            else:
                st.warning(f"⚠️ Chỉ {passed_count}/4 yếu tố — chưa đủ điều kiện vào lệnh BUY.")

            st.warning(
                """
**Lỗi phổ biến của người mới:**

❌ Vào lệnh khi RSI oversold nhưng trend vẫn downtrend mạnh  
→ *"Dao rơi đừng hứng"* — giá oversold vẫn có thể giảm tiếp

❌ Bỏ qua volume — tín hiệu không có volume xác nhận thường là tín hiệu giả

❌ Không kiểm tra khung lớn — 15m BUY nhưng 1d SELL mạnh = bẫy
"""
            )

        with tab4:
            st.markdown(
                f"""
<div style="font-size:14px;">

## Vào lệnh, SL và TP như thế nào?

### Entry — Điểm vào lệnh:
📌 **Market Order:** mua/bán ngay giá hiện tại  
→ Dùng khi tín hiệu rất mạnh, sợ bỏ lỡ  
→ Nhược điểm: có thể bị slippage (trượt giá)

📌 **Limit Order:** đặt trước ở giá tốt hơn  
→ Dùng khi muốn mua thấp hơn 0.5-1% so với giá hiện tại  
→ Ưu điểm: giá vào tốt hơn, R:R cải thiện  
→ Nhược điểm: có thể không khớp nếu giá không về

### Stop Loss — Đặt ở đâu?
📏 **Quy tắc:** đặt ở mức mà nếu giá đến đó, thesis của bạn **SAI**

**Với BUY signal:**
→ SL dưới BB Lower 1.5-2%  
→ Hoặc dưới đáy nến gần nhất  
→ **KHÔNG** đặt SL quá gần (< 0.5%) — dễ bị hunt  
→ **KHÔNG** đặt SL quá xa (> 5%) — rủi ro quá lớn

### Take Profit — Chốt ở đâu?
🎯 **TP1:** BB Middle → chốt 50-60% vị thế (an toàn)  
🎯 **TP2:** BB Upper → để 40-50% còn lại chạy  
🎯 **TP3:** nếu trend rất mạnh → trailing stop

### Quản lý vốn — QUAN TRỌNG NHẤT:
💵 Mỗi lệnh chỉ rủi ro tối đa **1-2%** tổng vốn

**Ví dụ:**  
Vốn: **1,000$**  
Rủi ro mỗi lệnh: 1% = **10$**  
Entry: **{m['last_close']:,.2f}**, SL: **{m['sl_buy']:,.2f}**  
Khoảng cách SL: **{m['sl_pct']:.1f}%**  
→ Số lượng nên mua: **{m['position_size']:.4f}** đơn vị

### Đặt lệnh trên Binance — từng bước:
1️⃣ Vào Binance → chọn cặp (ví dụ BNB/USDT)  
2️⃣ Chọn tab **[Spot]** hoặc **[Futures]**  
3️⃣ Chọn **[Limit]** order  
4️⃣ Nhập giá Entry và số lượng  
5️⃣ Nhấn **[Buy BNB]** → xác nhận  
6️⃣ Sau khi khớp → vào **[Orders]** → đặt **OCO order**  
   OCO = Stop Loss + Take Profit cùng lúc  
   → Stop Price: **{m['sl_buy']:,.2f}** (SL)  
   → Limit Price: **{m['tp1_buy']:,.2f}** (TP1)  
7️⃣ Theo dõi và điều chỉnh nếu cần

</div>
""",
                unsafe_allow_html=True,
            )
            st.info(
                "💡 **Tip:** Luôn đặt OCO ngay sau khi lệnh khớp — "
                "đừng để lệnh mở mà không có SL."
            )

        st.warning(
            """
⚠️ **Disclaimer:** Đây là công cụ học tập và phân tích kỹ thuật.  
Không phải lời khuyên tài chính.  
Luôn quản lý rủi ro — chỉ dùng vốn bạn sẵn sàng mất.
"""
        )

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _chart_xaxis_nav() -> dict:
    return {
        "rangeslider": dict(visible=True, thickness=0.04),
        "rangeselector": dict(
            buttons=[
                dict(count=1, label="1D", step="day", stepmode="backward"),
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        ),
    }


CHART_THEME = {
    "font": dict(color="#24292f", size=13),
    "title_font": dict(color="#24292f", size=15),
    "legend": dict(
        font=dict(color="#24292f", size=12),
        bgcolor="rgba(246,248,250,0.8)",
        bordercolor="#d0d7de",
        borderwidth=1,
    ),
    "xaxis": dict(
        tickfont=dict(color="#57606a", size=12),
        title_font=dict(color="#57606a"),
        gridcolor="#eaeef2",
        linecolor="#d0d7de",
    ),
    "yaxis": dict(
        tickfont=dict(color="#57606a", size=12),
        title_font=dict(color="#57606a"),
        gridcolor="#eaeef2",
        linecolor="#d0d7de",
    ),
    "paper_bgcolor": "#ffffff",
    "plot_bgcolor": "#ffffff",
}


def _apply_chart_layout(
    fig,
    title: str,
    height: int,
    legend_extra: dict | None = None,
    yaxis_extra: dict | None = None,
    showlegend: bool = True,
) -> None:
    xaxis = {**CHART_THEME["xaxis"], **_chart_xaxis_nav()}
    yaxis = {**CHART_THEME["yaxis"], **(yaxis_extra or {})}

    layout = {
        "template": "plotly_white",
        "font": CHART_THEME["font"],
        "title": title,
        "title_font": CHART_THEME["title_font"],
        "paper_bgcolor": CHART_THEME["paper_bgcolor"],
        "plot_bgcolor": CHART_THEME["plot_bgcolor"],
        "xaxis": xaxis,
        "yaxis": yaxis,
        "height": height,
        "showlegend": showlegend,
        "dragmode": "pan",
        "modebar_add": ["v1hovermode", "toggleSpikelines"],
    }

    if showlegend:
        layout["legend"] = {**CHART_THEME["legend"], **(legend_extra or {})}

    fig.update_layout(**layout)


def build_price_chart(df):
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["open_time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["ma_7"],
            mode="lines",
            line=dict(color="yellow", width=1.5),
            name="MA 7",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["ma_25"],
            mode="lines",
            line=dict(color="orange", width=1.5),
            name="MA 25",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["ma_99"],
            mode="lines",
            line=dict(color="red", width=1.5),
            name="MA 99",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["bb_upper"],
            mode="lines",
            line=dict(color="blue", dash="dash", width=1),
            name="BB Upper",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["bb_middle"],
            mode="lines",
            line=dict(color="grey", dash="dash", width=1),
            name="BB Middle",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=df["bb_lower"],
            mode="lines",
            line=dict(color="blue", dash="dash", width=1),
            name="BB Lower",
        )
    )

    _apply_chart_layout(
        fig,
        title="Candlestick + Moving Averages + Bollinger Bands",
        height=500,
        legend_extra=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_volume_chart(df):
    colors = ["green" if close >= open_ else "red" for close, open_ in zip(df["close"], df["open"])]

    fig = go.Figure(
        go.Bar(
            x=df["open_time"],
            y=df["volume"],
            marker_color=colors,
            name="Volume",
        )
    )
    _apply_chart_layout(
        fig,
        title="Volume",
        height=250,
        showlegend=False,
    )
    return fig


def build_rsi_chart(df):
    fig = go.Figure()

    rsi = df["rsi"]
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=rsi.where(rsi > 70),
            mode="lines",
            line=dict(color="red", width=2),
            connectgaps=False,
            name="RSI > 70",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=rsi.where(rsi < 30),
            mode="lines",
            line=dict(color="green", width=2),
            connectgaps=False,
            name="RSI < 30",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["open_time"],
            y=rsi.where((rsi >= 30) & (rsi <= 70)),
            mode="lines",
            line=dict(color="#57606a", width=2),
            connectgaps=False,
            name="RSI",
        )
    )

    fig.add_hline(y=70, line=dict(color="red", dash="dash", width=1))
    fig.add_hline(y=30, line=dict(color="green", dash="dash", width=1))

    _apply_chart_layout(
        fig,
        title="RSI (14)",
        height=250,
        yaxis_extra=dict(range=[0, 100]),
        legend_extra=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Sidebar
st.sidebar.markdown("**Symbol**")
symbols = get_available_symbols()
symbol = st.sidebar.selectbox("Symbol", symbols or ["—"], label_visibility="collapsed")
st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown("**Interval**")
interval = st.sidebar.selectbox("Interval", config.TIMEFRAMES, label_visibility="collapsed")
st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown("**Candles to display**")
limit = st.sidebar.slider(
    "Candles to display",
    min_value=50,
    max_value=500,
    value=200,
    label_visibility="collapsed",
)

st.title("Crypto Market Intelligence Dashboard")

all_signals = load_signals()

# Row 1: Signal Summary (40%) | MTF Conclusion (60%)
col_left, col_right = st.columns([4, 6])

with col_left:
    st.markdown("### Signal Summary")
    if symbols:
        st.markdown(
            build_signal_table_html(all_signals, symbols),
            unsafe_allow_html=True,
        )
    else:
        st.info("No symbols in database. Run `python main.py` first.")

with col_right:
    render_mtf_conclusion(symbols)

# Row 2+: selected symbol detail & charts
if not symbols:
    st.stop()

df = load_candles(symbol, interval, limit)

if df.empty:
    st.warning(
        f"No data found for {symbol} ({interval}). Run `python main.py` first to fetch and store candle data."
    )
    st.stop()

df = add_all_indicators(df)

render_price_action_analysis(df, symbol, interval)
render_backtesting(symbol, interval)
render_analysis_guide(df, symbol, interval)

st.plotly_chart(build_price_chart(df), use_container_width=True)
st.plotly_chart(build_volume_chart(df), use_container_width=True)
st.plotly_chart(build_rsi_chart(df), use_container_width=True)
