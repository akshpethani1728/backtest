"""
Backtesting Application - Streamlit Version
EMA Crossover with ATR Trailing Stop Loss Strategy
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Backtesting Tool",
    page_icon="📈",
    layout="wide"
)

# =============================================================================
# DATA FETCHING
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_data(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for '{symbol}'. Check symbol and date range.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        df = df[df['Close'] > 0]

        return df

    except Exception as e:
        raise ValueError(str(e))

# =============================================================================
# INDICATORS
# =============================================================================

def calculate_indicators(df: pd.DataFrame, ema_fast: int = 20, ema_slow: int = 50, atr_period: int = 14) -> pd.DataFrame:
    """Calculate EMA 20, EMA 50, and ATR indicators."""
    df = df.copy()

    df['EMA_Fast'] = df['Close'].ewm(span=ema_fast, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=ema_slow, adjust=False).mean()

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=atr_period).mean()

    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def run_backtest(df: pd.DataFrame, atr_multiplier: float = 1.5) -> list:
    """
    Run backtesting engine with EMA crossover + ATR trailing stop strategy.
    """
    df = df.copy()
    df = calculate_indicators(df)

    trades = []
    position = None
    entry_price = 0
    entry_time = None
    atr_trailing_stop = 0
    in_pullback = False
    prev_ema_fast = None
    prev_ema_slow = None

    for i, (idx, row) in enumerate(df.iterrows()):
        if i < max(50, 14):
            prev_ema_fast = row['EMA_Fast']
            prev_ema_slow = row['EMA_Slow']
            continue

        curr_ema_fast = row['EMA_Fast']
        curr_ema_slow = row['EMA_Slow']
        curr_close = row['Close']
        curr_high = row['High']
        curr_low = row['Low']
        curr_atr = row['ATR']

        prev_crossover_up = (prev_ema_fast > prev_ema_slow and curr_ema_fast > curr_ema_slow)
        prev_crossover_down = (prev_ema_fast < prev_ema_slow and curr_ema_fast < curr_ema_slow)

        curr_crossover_up = (prev_ema_fast <= prev_ema_slow and curr_ema_fast > curr_ema_slow)
        curr_crossover_down = (prev_ema_fast >= prev_ema_slow and curr_ema_fast < curr_ema_slow)

        # === CLOSE EXISTING POSITION ===
        if position == 'buy':
            new_stop = curr_close - (atr_multiplier * curr_atr)
            if new_stop > atr_trailing_stop:
                atr_trailing_stop = new_stop

            if curr_low <= atr_trailing_stop:
                trades.append({
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'entry_price': round(entry_price, 5),
                    'exit_time': idx.strftime('%Y-%m-%d %H:%M'),
                    'exit_price': round(atr_trailing_stop, 5),
                    'direction': 'BUY',
                    'pnl': round(atr_trailing_stop - entry_price, 5),
                    'pnl_pct': round(((atr_trailing_stop - entry_price) / entry_price) * 100, 2),
                    'exit_reason': 'Stop Loss'
                })
                position = None
                atr_trailing_stop = 0
                in_pullback = False

            elif curr_crossover_down and curr_close < curr_ema_fast and curr_close < curr_ema_slow:
                trades.append({
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'entry_price': round(entry_price, 5),
                    'exit_time': idx.strftime('%Y-%m-%d %H:%M'),
                    'exit_price': round(curr_close, 5),
                    'direction': 'BUY',
                    'pnl': round(curr_close - entry_price, 5),
                    'pnl_pct': round(((curr_close - entry_price) / entry_price) * 100, 2),
                    'exit_reason': 'Opposite Signal'
                })
                position = None
                atr_trailing_stop = 0
                in_pullback = False

        elif position == 'sell':
            new_stop = curr_close + (atr_multiplier * curr_atr)
            if new_stop < atr_trailing_stop:
                atr_trailing_stop = new_stop

            if curr_high >= atr_trailing_stop:
                trades.append({
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'entry_price': round(entry_price, 5),
                    'exit_time': idx.strftime('%Y-%m-%d %H:%M'),
                    'exit_price': round(atr_trailing_stop, 5),
                    'direction': 'SELL',
                    'pnl': round(entry_price - atr_trailing_stop, 5),
                    'pnl_pct': round(((entry_price - atr_trailing_stop) / entry_price) * 100, 2),
                    'exit_reason': 'Stop Loss'
                })
                position = None
                atr_trailing_stop = 0
                in_pullback = False

            elif curr_crossover_up and curr_close > curr_ema_fast and curr_close > curr_ema_slow:
                trades.append({
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'entry_price': round(entry_price, 5),
                    'exit_time': idx.strftime('%Y-%m-%d %H:%M'),
                    'exit_price': round(curr_close, 5),
                    'direction': 'SELL',
                    'pnl': round(entry_price - curr_close, 5),
                    'pnl_pct': round(((entry_price - curr_close) / entry_price) * 100, 2),
                    'exit_reason': 'Opposite Signal'
                })
                position = None
                atr_trailing_stop = 0
                in_pullback = False

        # === NEW ENTRY ===
        if position is None:
            # BUY SIGNAL
            if curr_crossover_up and curr_close > curr_ema_fast and curr_close > curr_ema_slow:
                in_pullback = True
                pullback_crossover_idx = i

            elif in_pullback:
                pullback_tolerance = curr_atr * 0.5
                near_ema = curr_low <= (curr_ema_fast + pullback_tolerance) and curr_low >= (curr_ema_fast - pullback_tolerance * 3)

                if near_ema and curr_close > curr_ema_fast and curr_close > curr_ema_slow:
                    position = 'buy'
                    entry_price = curr_close
                    entry_time = idx
                    atr_trailing_stop = curr_close - (atr_multiplier * curr_atr)
                    in_pullback = False
                elif curr_close < curr_ema_slow:
                    in_pullback = False

            # SELL SIGNAL
            if position is None and curr_crossover_down and curr_close < curr_ema_fast and curr_close < curr_ema_slow:
                in_pullback = True
                pullback_crossover_idx = i

            elif in_pullback and position is None:
                pullback_tolerance = curr_atr * 0.5
                near_ema = curr_high >= (curr_ema_fast - pullback_tolerance) and curr_high <= (curr_ema_fast + pullback_tolerance * 3)

                if near_ema and curr_close < curr_ema_fast and curr_close < curr_ema_slow:
                    position = 'sell'
                    entry_price = curr_close
                    entry_time = idx
                    atr_trailing_stop = curr_close + (atr_multiplier * curr_atr)
                    in_pullback = False
                elif curr_close > curr_ema_slow:
                    in_pullback = False

        prev_ema_fast = curr_ema_fast
        prev_ema_slow = curr_ema_slow

    # Close open position at end
    if position is not None and len(df) > 0:
        last_close = df.iloc[-1]['Close']
        if position == 'buy':
            trades.append({
                'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                'entry_price': round(entry_price, 5),
                'exit_time': df.index[-1].strftime('%Y-%m-%d %H:%M'),
                'exit_price': round(last_close, 5),
                'direction': 'BUY',
                'pnl': round(last_close - entry_price, 5),
                'pnl_pct': round(((last_close - entry_price) / entry_price) * 100, 2),
                'exit_reason': 'End of Data'
            })
        else:
            trades.append({
                'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                'entry_price': round(entry_price, 5),
                'exit_time': df.index[-1].strftime('%Y-%m-%d %H:%M'),
                'exit_price': round(last_close, 5),
                'direction': 'SELL',
                'pnl': round(entry_price - last_close, 5),
                'pnl_pct': round(((entry_price - last_close) / entry_price) * 100, 2),
                'exit_reason': 'End of Data'
            })

    return trades

# =============================================================================
# SUMMARY CALCULATIONS
# =============================================================================

def calculate_summary(trades: list) -> dict:
    """Calculate summary statistics."""
    if not trades:
        return {
            'total_trades': 0, 'win_rate': 0, 'net_profit': 0,
            'avg_profit': 0, 'max_drawdown': 0, 'buy_trades': 0,
            'sell_trades': 0, 'winning_trades': 0, 'losing_trades': 0
        }

    total = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    net_profit = sum(t['pnl'] for t in trades)
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cumulative += t['pnl']
        if cumulative > peak:
            peak = cumulative
        drawdown = peak - cumulative
        if drawdown > max_dd:
            max_dd = drawdown

    return {
        'total_trades': total,
        'win_rate': round((len(wins) / total) * 100, 2) if total > 0 else 0,
        'net_profit': round(net_profit, 5),
        'avg_profit': round(net_profit / total, 5) if total > 0 else 0,
        'max_drawdown': round(max_dd, 5),
        'buy_trades': len([t for t in trades if t['direction'] == 'BUY']),
        'sell_trades': len([t for t in trades if t['direction'] == 'SELL']),
        'winning_trades': len(wins),
        'losing_trades': len(losses)
    }

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.title("📈 Backtesting Tool")
st.caption("EMA Crossover + ATR Trailing Stop Strategy")

# Sidebar inputs
st.sidebar.header("Settings")

symbol = st.sidebar.text_input("Symbol", value="EURUSD=X", help="Yahoo Finance format: EURUSD=X, BTC-USD, AAPL")
timeframe = st.sidebar.selectbox("Timeframe", ["15m", "1h", "1d"], index=1)

today = datetime.today()
default_start = datetime(today.year - 1, today.month, today.day)

start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=today)

atr_multiplier = st.sidebar.number_input("ATR Multiplier", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

# Example symbols
st.sidebar.markdown("---")
st.sidebar.markdown("**Example Symbols:**")
st.sidebar.markdown("""
- Forex: `EURUSD=X`, `GBPUSD=X`
- Crypto: `BTC-USD`, `ETH-USD`
- Stocks: `AAPL`, `TSLA`
""")

# Run button
run = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)

if run:
    # Validation
    if not symbol.strip():
        st.error("Please enter a symbol.")
        st.stop()

    if start_date >= end_date:
        st.error("End date must be after start date.")
        st.stop()

    delta = (end_date - start_date).days
    if timeframe == '15m' and delta > 60:
        st.error("For 15m data, date range cannot exceed 60 days.")
        st.stop()
    elif timeframe == '1h' and delta > 730:
        st.error("For hourly data, date range cannot exceed 2 years.")
        st.stop()

    with st.spinner("Fetching data..."):
        try:
            df = fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), timeframe)
        except Exception as e:
            st.error(f"Data Error: {str(e)}")
            st.stop()

    if len(df) < 100:
        st.error(f"Not enough data. Got {len(df)} candles. Need at least 100.")
        st.stop()

    with st.spinner("Running backtest..."):
        trades = run_backtest(df, atr_multiplier)
        summary = calculate_summary(trades)

    # Results
    st.markdown("---")
    st.subheader("📊 Summary")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Trades", summary['total_trades'])
    col2.metric("Win Rate", f"{summary['win_rate']}%", delta_color="normal" if summary['win_rate'] >= 50 else "inverse")
    col3.metric("Net Profit", f"{summary['net_profit']:.5f}", delta_color="normal" if summary['net_profit'] >= 0 else "inverse")
    col4.metric("Avg Profit", f"{summary['avg_profit']:.5f}")
    col5.metric("Max Drawdown", f"{summary['max_drawdown']:.5f}")

    col6, col7, col8, col9 = st.columns(4)
    col6.metric("Buy Trades", summary['buy_trades'])
    col7.metric("Sell Trades", summary['sell_trades'])
    col8.metric("Winning", summary['winning_trades'])
    col9.metric("Losing", summary['losing_trades'])

    # Equity Curve
    if trades:
        st.markdown("---")
        st.subheader("📈 Equity Curve")

        equity_curve = []
        cumulative = 0
        for trade in trades:
            cumulative += trade['pnl']
            equity_curve.append({'time': trade['exit_time'], 'equity': cumulative})

        equity_df = pd.DataFrame(equity_curve)

        fig = px.line(equity_df, x='time', y='equity', title='Cumulative P&L',
                      labels={'time': 'Date', 'equity': 'Equity'})
        fig.update_layout(template='plotly_white', hovermode='x unified')
        fig.update_traces(line_color='#1a1a2e', line_width=2)
        st.plotly_chart(fig, use_container_width=True)

    # Trade Table
    if trades:
        st.markdown("---")
        st.subheader("📋 Trade History")

        trades_df = pd.DataFrame(trades)
        trades_df.index = range(1, len(trades_df) + 1)
        trades_df.index.name = '#'

        st.dataframe(
            trades_df.style.applymap(
                lambda x: 'color: #28a745' if isinstance(x, (int, float)) and x > 0 else ('color: #dc3545' if isinstance(x, (int, float)) and x < 0 else ''),
                subset=['pnl', 'pnl_pct']
            ),
            use_container_width=True
        )
    else:
        st.info("No trades generated with the current parameters.")

else:
    st.info("👈 Configure settings in the sidebar and click **Run Backtest** to start.")
