# app.py

import os
import datetime
import pytz
import requests
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import streamlit as st
import plotly.graph_objects as go

# =========================
# Page / Config
# =========================
st.set_page_config(page_title="Real-time Stock Tracker & Backtester", layout="wide")
st.title("📈 Real-time Stock Tracker, Alerts & Backtester")

# =========================
# Secrets (Telegram)
# =========================
BOT_TOKEN = st.secrets.get("BOT_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

def send_telegram_message(message: str):
    if not BOT_TOKEN or not CHAT_ID:
        st.warning("Telegram secrets missing. Add BOT_TOKEN and CHAT_ID in secrets.")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        st.error(f"Error sending Telegram message: {e}")

# =========================
# Utility
# =========================
def is_market_open() -> bool:
    tz = pytz.timezone("US/Eastern")
    now = datetime.datetime.now(tz)
    if now.weekday() >= 5:  # 5=Sat,6=Sun
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close

def valid_ema_periods(short_p: int, long_p: int) -> tuple[int, int]:
    # Ensure short < long; if not, swap them
    if short_p >= long_p:
        short_p, long_p = long_p - 1, long_p
        short_p = max(5, short_p)
    return short_p, long_p

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.header("⚙️ Controls")
    symbols_input = st.text_input("Stock Symbols (comma separated)", value="AAPL, MSFT, TSLA")
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

    colp1, colp2 = st.columns(2)
    with colp1:
        period = st.selectbox(
            "Data Period",
            ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
            index=1
        )
    with colp2:
        interval = st.selectbox(
            "Interval",
            ["5m", "15m", "30m", "1h", "1d", "1wk", "1mo"],
            index=3
        )

    colE1, colE2 = st.columns(2)
    with colE1:
        ema_short = st.number_input("Short EMA", min_value=5, max_value=100, value=20, step=1)
    with colE2:
        ema_long = st.number_input("Long EMA", min_value=10, max_value=300, value=50, step=1)

    use_rsi = st.checkbox("Use RSI filter (BUY if RSI<30, SELL if RSI>70)", value=True)

    only_market_hours = st.checkbox("Only run signals during market hours", value=True)

    st.caption(
        "Note: Yahoo Finance restricts some period/interval combos (e.g., very small intervals with very long periods). "
        "If data returns empty, try a shorter period or larger interval."
    )

# Ensure EMA periods make sense
ema_short, ema_long = valid_ema_periods(int(ema_short), int(ema_long))
short_col = f"EMA{ema_short}"
long_col = f"EMA{ema_long}"

# =========================
# Session State
# =========================
if "sent_signals" not in st.session_state:
    st.session_state.sent_signals = {}  # {symbol: last_signal_text}

# =========================
# Data + Indicators
# =========================
def load_and_update_data(symbol: str, period: str, interval: str, ema_short: int, ema_long: int) -> pd.DataFrame | None:
    """
    Loads existing CSV if present, downloads new data, merges, computes indicators,
    drops NaNs, and writes back to CSV.
    """
    csv_filename = f"{symbol}_data.csv"
    if os.path.exists(csv_filename):
        try:
            df_existing = pd.read_csv(csv_filename, index_col=0, parse_dates=True)
        except Exception:
            df_existing = pd.DataFrame()
    else:
        df_existing = pd.DataFrame()

    # Download new data
    df_new = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df_new is None or df_new.empty:
        return None

    # Merge, drop duplicates
    df = pd.concat([df_existing, df_new])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Indicators
    df[short_col] = df["Close"].ewm(span=ema_short, adjust=False).mean()
    df[long_col] = df["Close"].ewm(span=ema_long, adjust=False).mean()
    df["RSI"] = ta.rsi(df["Close"], length=14)

    df.dropna(inplace=True)

    # Save back to CSV
    try:
        df.to_csv(csv_filename)
    except Exception as e:
        st.warning(f"Could not save CSV for {symbol}: {e}")

    return df

# =========================
# Signals
# =========================
def latest_signal(df: pd.DataFrame, use_rsi: bool) -> str | None:
    if len(df) < 2:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]

    signal_parts = []

    # EMA cross
    bullish_cross = (prev[short_col] < prev[long_col]) and (last[short_col] > last[long_col])
    bearish_cross = (prev[short_col] > prev[long_col]) and (last[short_col] < last[long_col])

    # RSI conditions (if toggled)
    rsi_buy_ok = (last["RSI"] < 30) if use_rsi else True
    rsi_sell_ok = (last["RSI"] > 70) if use_rsi else True

    if bullish_cross and rsi_buy_ok:
        signal_parts.append("Bullish crossover → Buying opportunity")
    if bearish_cross and rsi_sell_ok:
        signal_parts.append("Bearish crossover → Selling opportunity")

    # Informational RSI-only notes
    if last["RSI"] > 70:
        signal_parts.append("RSI overbought")
    elif last["RSI"] < 30:
        signal_parts.append("RSI oversold")

    if not signal_parts:
        return None

    return " | ".join(signal_parts)

# =========================
# Backtest
# =========================
def backtest_strategy(
    df: pd.DataFrame,
    ema_short: int,
    ema_long: int,
    use_rsi: bool = False,
    initial_capital: float = 10_000.0
):
    """
    Simple long-only, all-in/all-out backtest:
    - Buy at next bar's Open on bullish cross (and RSI<30 if toggled)
    - Sell at next bar's Open on bearish cross (and RSI>70 if toggled)
    """
    cash = initial_capital
    shares = 0
    trades = []

    equity_curve = []  # (timestamp, equity)
    idx = df.index

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        bullish_cross = (prev[short_col] < prev[long_col]) and (curr[short_col] > curr[long_col])
        bearish_cross = (prev[short_col] > prev[long_col]) and (curr[short_col] < curr[long_col])

        rsi_buy_ok = (curr["RSI"] < 30) if use_rsi else True
        rsi_sell_ok = (curr["RSI"] > 70) if use_rsi else True

        # BUY
        if bullish_cross and shares == 0 and rsi_buy_ok:
            # Buy at current bar Open
            price = float(curr["Open"])
            if price > 0:
                shares = int(cash // price)
                cost = shares * price
                cash -= cost
                trades.append(("BUY", idx[i], price, shares))

        # SELL
        elif bearish_cross and shares > 0 and rsi_sell_ok:
            price = float(curr["Open"])
            cash += shares * price
            trades.append(("SELL", idx[i], price, shares))
            shares = 0

        # Track equity at the current bar's Close
        equity = cash + shares * float(curr["Close"])
        equity_curve.append((idx[i], equity))

    # Close any open position at final close (mark-to-market already in equity)
    final_equity = equity_curve[-1][1] if equity_curve else initial_capital
    profit_pct = (final_equity - initial_capital) / initial_capital * 100.0

    # Win rate calc
    wins = 0
    closed_trades = 0
    # Pair BUY/SELL to compute PnL
    entry_price = None
    for action, ts, price, qty in trades:
        if action == "BUY":
            entry_price = price
        elif action == "SELL" and entry_price is not None:
            pnl = price - entry_price
            wins += 1 if pnl > 0 else 0
            closed_trades += 1
            entry_price = None

    win_rate = (wins / closed_trades * 100.0) if closed_trades > 0 else 0.0

    # Max drawdown from equity curve
    max_dd = 0.0
    peak = -float("inf")
    for _, eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = max_dd * 100.0

    # Equity DataFrame for plotting
    eq_df = pd.DataFrame(equity_curve, columns=["Time", "Equity"]).set_index("Time")

    return {
        "final_value": final_equity,
        "profit_pct": profit_pct,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "max_drawdown_pct": max_dd_pct,
        "trades": trades,
        "equity_df": eq_df,
    }

# =========================
# Main Loop per Symbol
# =========================
for symbol in symbols:
    st.markdown(f"### 🧾 {symbol}")

    if only_market_hours and not is_market_open():
        st.warning("Market is closed. Live signals/alerts are paused (9:30 AM – 4:00 PM ET). You can still backtest and chart.")
        # We still allow fetching historical for chart/backtest

    df = load_and_update_data(symbol, period=period, interval=interval, ema_short=ema_short, ema_long=ema_long)
    if df is None or df.empty:
        st.error(f"No data returned for {symbol} with period='{period}' & interval='{interval}'. Try a different combo.")
        continue

    # ---------------------
    # Live Signal + Alert
    # ---------------------
    sig = latest_signal(df, use_rsi=use_rsi)
    if sig:
        st.info(f"**Latest Signal:** {sig}")
        # Send Telegram only if:
        if (not only_market_hours) or is_market_open():
            last_sig = st.session_state.sent_signals.get(symbol)
            if sig != last_sig:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                send_telegram_message(f"{symbol} - {sig} at {timestamp} ET")
                st.session_state.sent_signals[symbol] = sig
    else:
        st.write("No new trade signal at the latest bar.")

    # ---------------------
    # Price Chart
    # ---------------------
    with st.container():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
        fig.add_trace(go.Scatter(x=df.index, y=df[short_col], mode="lines", name=short_col))
        fig.add_trace(go.Scatter(x=df.index, y=df[long_col], mode="lines", name=long_col))
        fig.update_layout(
            title=f"{symbol} Price with {short_col} & {long_col}",
            xaxis_title="Time",
            yaxis_title="Price",
            margin=dict(l=10, r=10, t=35, b=10),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------
    # Backtest (on full CSV history)
    # ---------------------
    results = backtest_strategy(df, ema_short=ema_short, ema_long=ema_long, use_rsi=use_rsi, initial_capital=10_000.0)

    colm1, colm2, colm3, colm4 = st.columns(4)
    colm1.metric("Final Value", f"${results['final_value']:.2f}")
    colm2.metric("Total Return", f"{results['profit_pct']:.2f}%")
    colm3.metric("Win Rate", f"{results['win_rate']:.1f}%")
    colm4.metric("Max Drawdown", f"{results['max_drawdown_pct']:.1f}%")

    # Equity curve chart
    if results["equity_df"] is not None and not results["equity_df"].empty:
        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(x=results["equity_df"].index, y=results["equity_df"]["Equity"], mode="lines", name="Equity"))
        eq_fig.update_layout(
            title=f"{symbol} Backtest Equity Curve",
            xaxis_title="Time",
            yaxis_title="Equity ($)",
            margin=dict(l=10, r=10, t=35, b=10),
            height=300
        )
        st.plotly_chart(eq_fig, use_container_width=True)

    # Trades
    with st.expander("📝 Trade Log"):
        if results["trades"]:
            trades_df = pd.DataFrame(results["trades"], columns=["Action", "Time", "Price", "Shares"])
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.write("No trades executed under current settings.")

    # Last 5 rows
    with st.expander("🔎 Latest Data (last 5 rows)"):
        st.dataframe(df.tail(5), use_container_width=True)
