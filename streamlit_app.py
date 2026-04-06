"""
Backtesting Application - Streamlit Version
EMA Crossover with ATR Trailing Stop Strategy
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Backtesting Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark theme CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Text colors */
    .stApp, .stApp p, .stApp span {
        color: #c9d1d9;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
    }

    /* Metric value */
    div[data-testid="metric-container"] label {
        color: #8b949e !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #161b22;
        border-radius: 8px;
    }

    /* Tabs */
    .stTabs [data-selected="true"] {
        background-color: #238636;
    }

    /* Dataframe */
    .dataframe {
        background-color: #161b22 !important;
    }

    /* Success/positive values */
    .positive {
        color: #3fb950 !important;
    }

    /* Danger/negative values */
    .negative {
        color: #f85149 !important;
    }

    /* Hide hamburger menu */
    #MainMenu {visibility: hidden;}

    /* Footer */
    footer {visibility: hidden;}

    /* Custom divider */
    hr {
        border-color: #30363d;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA FETCHING
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance using period parameter."""
    try:
        ticker = yf.Ticker(symbol)

        # Map interval to yfinance period (yfinance uses period for auto DateRange)
        # period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        period_map = {
            '1 Month': '1mo',
            '3 Months': '3mo',
            '6 Months': '6mo',
            '1 Year': '1y',
            'Max Available': '2y'  # Use 2y as reasonable max for most assets
        }
        yf_period = period_map.get(period, '1y')

        df = ticker.history(period=yf_period, interval=interval)

        if df.empty:
            raise ValueError(f"No data available for '{symbol}'. Try a different symbol or timeframe.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required data column: {col}")

        # Drop rows with missing essential data
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        df = df[df['Close'] > 0]

        if df.empty:
            raise ValueError(f"No valid data for '{symbol}'. Try a different symbol or timeframe.")

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

    min_periods = max(ema_fast, ema_slow, atr_period)
    df = df.iloc[min_periods:]
    df = df.dropna(subset=['EMA_Fast', 'EMA_Slow', 'ATR', 'Open', 'High', 'Low', 'Close'])

    if df.empty:
        raise ValueError("Not enough data to calculate indicators. Try a different timeframe or period.")

    return df

# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def run_backtest(df: pd.DataFrame, atr_multiplier: float = 1.5) -> list:
    """Run backtesting engine with EMA crossover + ATR trailing stop strategy."""
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
                    'exit_time': idx.strftime('%Y-%m-%d %H:%M'),
                    'direction': 'BUY',
                    'entry_price': round(entry_price, 5),
                    'exit_price': round(atr_trailing_stop, 5),
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
                    'exit_time': idx.strftime('%Y-%m-%d %H:%M'),
                    'direction': 'BUY',
                    'entry_price': round(entry_price, 5),
                    'exit_price': round(curr_close, 5),
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
                    'exit_time': idx.strftime('%Y-%m-%d %H:%M'),
                    'direction': 'SELL',
                    'entry_price': round(entry_price, 5),
                    'exit_price': round(atr_trailing_stop, 5),
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
                    'exit_time': idx.strftime('%Y-%m-%d %H:%M'),
                    'direction': 'SELL',
                    'entry_price': round(entry_price, 5),
                    'exit_price': round(curr_close, 5),
                    'pnl': round(entry_price - curr_close, 5),
                    'pnl_pct': round(((entry_price - curr_close) / entry_price) * 100, 2),
                    'exit_reason': 'Opposite Signal'
                })
                position = None
                atr_trailing_stop = 0
                in_pullback = False

        # === NEW ENTRY ===
        if position is None:
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
        direction = 'BUY' if position == 'buy' else 'SELL'
        pnl = last_close - entry_price if position == 'buy' else entry_price - last_close
        trades.append({
            'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
            'exit_time': df.index[-1].strftime('%Y-%m-%d %H:%M'),
            'direction': direction,
            'entry_price': round(entry_price, 5),
            'exit_price': round(last_close, 5),
            'pnl': round(pnl, 5),
            'pnl_pct': round((pnl / entry_price) * 100, 2),
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
            'sell_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'largest_win': 0, 'largest_loss': 0
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

    largest_win = max([t['pnl'] for t in wins]) if wins else 0
    largest_loss = min([t['pnl'] for t in losses]) if losses else 0

    return {
        'total_trades': total,
        'win_rate': round((len(wins) / total) * 100, 2) if total > 0 else 0,
        'net_profit': round(net_profit, 5),
        'avg_profit': round(net_profit / total, 5) if total > 0 else 0,
        'max_drawdown': round(max_dd, 5),
        'buy_trades': len([t for t in trades if t['direction'] == 'BUY']),
        'sell_trades': len([t for t in trades if t['direction'] == 'SELL']),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'largest_win': round(largest_win, 5),
        'largest_loss': round(largest_loss, 5)
    }

# =============================================================================
# STREAMLIT UI
# =============================================================================

# Header
st.title("📈 Backtesting Tool")
st.caption("EMA Crossover + ATR Trailing Stop Strategy")

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("⚙️ Settings")

    # Symbol selection
    st.subheader("Symbol")
    symbol_options = {
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'BTC/USD': 'BTC-USD',
        'Apple': 'AAPL'
    }
    symbol_display = st.selectbox(
        "Select Asset",
        options=list(symbol_options.keys()),
        index=0,
        help="Choose from popular forex, crypto, or stock symbols"
    )
    symbol = symbol_options[symbol_display]

    st.divider()

    # Timeframe
    st.subheader("Timeframe")
    timeframe = st.selectbox(
        "Data Interval",
        options=['15m', '1h', '1d'],
        index=1,
        help="15m = 15 minutes, 1h = 1 hour, 1d = 1 day"
    )

    st.divider()

    # Period
    st.subheader("Date Range")
    period = st.selectbox(
        "Period",
        options=['1 Month', '3 Months', '6 Months', '1 Year', 'Max Available'],
        index=3,
        help="How much historical data to fetch"
    )

    st.divider()

    # ATR Multiplier
    st.subheader("Risk Management")
    atr_multiplier = st.slider(
        "ATR Multiplier",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1,
        help="Stop loss distance in ATR units. Higher = wider stop."
    )

    st.divider()

    # Run button
    run = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

# =============================================================================
# MAIN CONTENT
# =============================================================================

if run:
    # Fetch data
    with st.spinner(f"Fetching {symbol_display} data..."):
        try:
            df = fetch_data(symbol, period, timeframe)
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Failed to fetch data: {str(e)}")
            st.stop()

    if len(df) < 50:
        st.error(f"❌ Not enough data. Got {len(df)} candles. Try a longer period or different timeframe.")
        st.stop()

    # Run backtest
    with st.spinner("Running backtest..."):
        try:
            trades = run_backtest(df, atr_multiplier)
            summary = calculate_summary(trades)
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Backtest error: {str(e)}")
            st.stop()

    # Display info
    st.info(f"📊 Analyzed {len(df)} candles from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} | Symbol: {symbol_display} | Timeframe: {timeframe}")

    st.divider()

    # =============================================================================
    # METRIC CARDS
    # =============================================================================

    st.subheader("📊 Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    # Total Trades
    col1.metric(
        "Total Trades",
        summary['total_trades'],
        help="Total number of completed trades"
    )

    # Win Rate
    win_rate_delta = summary['win_rate'] - 50
    col2.metric(
        "Win Rate",
        f"{summary['win_rate']}%",
        delta=f"{'+' if win_rate_delta >= 0 else ''}{win_rate_delta}%" if summary['total_trades'] > 0 else None,
        delta_color="normal" if summary['win_rate'] >= 50 else "inverse",
        help="Percentage of winning trades"
    )

    # Net Profit
    profit_color = "normal" if summary['net_profit'] >= 0 else "inverse"
    col3.metric(
        "Net Profit",
        f"{summary['net_profit']:.5f}",
        delta_color=profit_color,
        help="Total profit/loss across all trades"
    )

    # Max Drawdown
    col4.metric(
        "Max Drawdown",
        f"{summary['max_drawdown']:.5f}",
        delta_color="inverse",
        help="Largest peak-to-trough decline"
    )

    col5, col6, col7, col8 = st.columns(4)

    col5.metric("Buy Trades", summary['buy_trades'])
    col6.metric("Sell Trades", summary['sell_trades'])
    col7.metric("Winning", summary['winning_trades'], delta_color="normal")
    col8.metric("Losing", summary['losing_trades'], delta_color="inverse")

    st.divider()

    # =============================================================================
    # EQUITY CURVE
    # =============================================================================

    st.subheader("📈 Equity Curve")

    if trades:
        equity_curve = []
        cumulative = 0
        for trade in trades:
            cumulative += trade['pnl']
            equity_curve.append({
                'time': trade['exit_time'],
                'equity': cumulative,
                'trade': f"{trade['direction']} #{len(equity_curve) + 1}"
            })

        equity_df = pd.DataFrame(equity_curve)

        # Create Plotly chart
        fig = go.Figure()

        # Add equity line
        fig.add_trace(go.Scatter(
            x=equity_df['time'],
            y=equity_df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#58a6ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(88, 166, 255, 0.1)'
        ))

        # Add markers for trades
        colors = ['#3fb950' if e >= 0 else '#f85149' for e in equity_df['equity']]
        fig.add_trace(go.Scatter(
            x=equity_df['time'],
            y=equity_df['equity'],
            mode='markers',
            name='Trade',
            marker=dict(color=colors, size=8),
            hoverinfo='text',
            hovertext=[f"Trade {i+1}: {e:.5f}" for i, e in enumerate(equity_df['equity'])]
        ))

        fig.update_layout(
            template='plotly_dark',
            hovermode='x unified',
            xaxis_title='Date',
            yaxis_title='Cumulative P&L',
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='#30363d',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='#30363d',
                showgrid=True
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades generated with the current parameters.")

    st.divider()

    # =============================================================================
    # TRADE HISTORY
    # =============================================================================

    st.subheader("📋 Trade History")

    if trades:
        trades_df = pd.DataFrame(trades)

        # Add row numbers
        trades_df.insert(0, '#', range(1, len(trades_df) + 1))

        # Style function for dataframe
        def style_pnl(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return 'color: #3fb950'
                elif val < 0:
                    return 'color: #f85149'
            return ''

        def style_direction(val):
            if val == 'BUY':
                return 'color: #3fb950; font-weight: bold'
            elif val == 'SELL':
                return 'color: #f85149; font-weight: bold'
            return ''

        # Apply styling
        styled_df = trades_df.style.applymap(style_pnl, subset=['pnl', 'pnl_pct'])
        styled_df = styled_df.applymap(style_direction, subset=['direction'])

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )

        # Download CSV
        csv = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade History (CSV)",
            data=csv,
            file_name=f"backtest_{symbol.replace('=', '_').replace('-', '_')}_{timeframe}.csv",
            mime="text/csv"
        )
    else:
        st.info("No trades to display.")

else:
    # Initial state - show instructions
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Strategy")
        st.markdown("""
        **Entry Rules (Buy):**
        1. EMA 20 crosses above EMA 50
        2. Price closes above both EMAs
        3. Wait for pullback to EMA 20
        4. Enter on confirmation candle

        **Entry Rules (Sell):**
        1. EMA 20 crosses below EMA 50
        2. Price closes below both EMAs
        3. Wait for pullback to EMA 20
        4. Enter on confirmation candle

        **Exit Rules:**
        - ATR trailing stop (moves only in profit direction)
        - Stop loss hit OR opposite signal appears
        """)

    with col2:
        st.subheader("📌 Available Symbols")
        st.markdown("""
        | Asset | Symbol |
        |-------|--------|
        | EUR/USD | EURUSD=X |
        | GBP/USD | GBPUSD=X |
        | BTC/USD | BTC-USD |
        | Apple | AAPL |

        *More symbols coming soon...*
        """)

    st.divider()

    st.info("👈 Configure settings in the sidebar and click **Run Backtest** to start.")
