import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import datetime
import requests
import pytz
import os

# ----------------------------
# Telegram Setup
# ----------------------------
BOT_TOKEN = st.secrets["BOT_TOKEN"]
CHAT_ID = st.secrets["CHAT_ID"]

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        st.error(f"Error sending Telegram message: {e}")

# ----------------------------
# Stock Analysis Functions
# ----------------------------
def get_stock_data(symbol, period="1mo", interval="1h"):
    df = yf.download(symbol, period=period, interval=interval)
    if df.empty:
        return None
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI"] = ta.rsi(df["Close"], length=14)
    df.dropna(inplace=True)
    return df

def check_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    signal = None
    # EMA crossover
    if prev["EMA20"] < prev["EMA50"] and last["EMA20"] > last["EMA50"]:
        signal = "Bullish crossover → Buying opportunity"
    elif prev["EMA20"] > prev["EMA50"] and last["EMA20"] < last["EMA50"]:
        signal = "Bearish crossover → Selling opportunity"
    
    # RSI filter
    if last["RSI"] > 70:
        signal = (signal + " | RSI overbought → consider selling") if signal else "RSI overbought → consider selling"
    elif last["RSI"] < 30:
        signal = (signal + " | RSI oversold → potential buying opportunity") if signal else "RSI oversold → potential buying opportunity"
    
    return signal

# ----------------------------
# RSI Divergence Detection
# ----------------------------
def check_rsi_divergence(df):
    """
    Detects bullish or bearish divergence
    """
    signal = None
    if len(df) < 3:
        return None

    closes = df['Close'].iloc[-3:].values
    rsi = df['RSI'].iloc[-3:].values

    # Bearish divergence: price makes higher high, RSI makes lower high
    if closes[2] > closes[1] > closes[0] and rsi[2] < rsi[1] < rsi[0]:
        signal = "⚠️ Bearish divergence detected: Price ↑, RSI ↓"

    # Bullish divergence: price makes lower low, RSI makes higher low
    elif closes[2] < closes[1] < closes[0] and rsi[2] > rsi[1] > rsi[0]:
        signal = "⚠️ Bullish divergence detected: Price ↓, RSI ↑"

    return signal

# ----------------------------
# Market Hours Checker
# ----------------------------
def is_market_open():
    tz = pytz.timezone("US/Eastern")
    now = datetime.datetime.now(tz)
    if now.weekday() >= 5:  # Sat or Sun
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Real-time Stock Tracker", layout="wide")
st.title("📈 Real-time Stock Tracker & Signal Bot")

# Input multiple stocks separated by commas
symbols_input = st.text_input("Enter Stock Symbols (comma separated)", value="AAPL, MSFT, TSLA")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

# Keep track of sent signals to prevent duplicates
if "sent_signals" not in st.session_state:
    st.session_state.sent_signals = {}

# Auto-refresh every 5 minutes
st_autorefresh_interval = 300000  # 300000 ms = 5 min
st.query_params.update({"refresh": [str(int(datetime.datetime.now().timestamp()))]})


# ----------------------------
# Main Loop
# ----------------------------
for symbol in symbols:
    st.subheader(f"Stock: {symbol}")

    if not is_market_open():
        st.warning("Market is closed. Updates will resume during market hours (9:30 AM – 4:00 PM ET).")
        continue

    df = get_stock_data(symbol, period="1mo", interval="1h")
    if df is None:
        st.error(f"No data found for {symbol}")
        continue

    # Save CSV
    csv_filename = f"{symbol}_data.csv"
    df.to_csv(csv_filename)

    # Check signals
    ema_rsi_signal = check_signal(df)
    divergence_signal = check_rsi_divergence(df)

    # Combine all signals
    full_signal = ema_rsi_signal
    if divergence_signal:
        full_signal = (full_signal + " | " + divergence_signal) if full_signal else divergence_signal

    st.write("Latest Signal:", full_signal)

    # Send Telegram if new signal
    last_signal = st.session_state.sent_signals.get(symbol)
    if full_signal and full_signal != last_signal:
        send_telegram_message(f"{symbol} - {full_signal} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.session_state.sent_signals[symbol] = full_signal

    # Display last 5 rows
    st.dataframe(df.tail(5))

    # Plot chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], mode='lines', name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], mode='lines', name="EMA50"))
    st.plotly_chart(fig, use_container_width=True)
