"""
Backtesting Application - EMA Crossover + ATR Trailing Stop
Enhanced with P&L in currency, more pairs, and autocomplete
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
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

# Dark theme CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3, h4 { color: #ffffff !important; }
    .stMarkdown, .stText { color: #c9d1d9; }
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
    }
    div[data-testid="metric-container"] label { color: #8b949e !important; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #ffffff !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    hr {border-color: #30363d;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SYMBOLS DATABASE
# =============================================================================

# Popular trading symbols organized by category
SYMBOLS = {
    'Forex': {
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'USD/JPY': 'USDJPY=X',
        'USD/CHF': 'USDCHF=X',
        'AUD/USD': 'AUDUSD=X',
        'USD/CAD': 'USDCAD=X',
        'NZD/USD': 'NZDUSD=X',
        'EUR/GBP': 'EURGBP=X',
        'EUR/JPY': 'EURJPY=X',
        'GBP/JPY': 'GBPJPY=X',
        'AUD/JPY': 'AUDJPY=X',
        'EUR/CHF': 'EURCHF=X',
    },
    'Crypto': {
        'BTC/USD': 'BTC-USD',
        'ETH/USD': 'ETH-USD',
        'SOL/USD': 'SOL-USD',
        'BNB/USD': 'BNB-USD',
        'XRP/USD': 'XRP-USD',
        'ADA/USD': 'ADA-USD',
        'DOGE/USD': 'DOGE-USD',
        'DOT/USD': 'DOT-USD',
        'AVAX/USD': 'AVAX-USD',
        'LINK/USD': 'LINK-USD',
        'MATIC/USD': 'MATIC-USD',
        'LTC/USD': 'LTC-USD',
    },
    'Stocks': {
        'Apple': 'AAPL',
        'Tesla': 'TSLA',
        'Microsoft': 'MSFT',
        'Google': 'GOOGL',
        'Amazon': 'AMZN',
        'NVIDIA': 'NVDA',
        'Meta': 'META',
        'Netflix': 'NFLX',
        'AMD': 'AMD',
        'Intel': 'INTC',
        'Gold': 'GC=F',
        'Oil': 'CL=F',
    }
}

# Build flat symbol map for autocomplete
ALL_SYMBOLS = {}
for category, pairs in SYMBOLS.items():
    for name, symbol in pairs.items():
        ALL_SYMBOLS[name] = symbol

SORTED_SYMBOLS = sorted(ALL_SYMBOLS.keys())

# =============================================================================
# DATA FETCHING
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch historical data using yfinance."""
    try:
        period_map = {
            '1 Month': '1mo',
            '3 Months': '3mo',
            '6 Months': '6mo',
            '1 Year': '1y',
            'Max Available': '2y'
        }
        yf_period = period_map.get(period, '1y')

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=yf_period, interval=interval)

        if df is None or df.empty:
            raise ValueError(f"No data available for '{symbol}'. Check the symbol.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ['Open', 'High', 'Low', 'Close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        df = df[df['Close'] > 0]

        if df.empty:
            raise ValueError(f"No valid price data for '{symbol}'.")

        return df

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to fetch data: {str(e)}")

# =============================================================================
# INDICATORS
# =============================================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate EMA 20, EMA 50, and ATR."""
    df = df.copy()

    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    df = df.dropna(subset=['EMA_20', 'EMA_50', 'ATR'])
    df = df[['Open', 'High', 'Low', 'Close', 'EMA_20', 'EMA_50', 'ATR']]

    if df.empty or len(df) < 10:
        raise ValueError("Not enough data. Try a longer period.")

    return df

# =============================================================================
# P&L CALCULATION
# =============================================================================

def calculate_pnl_money(pnl_points: float, symbol: str, lot_size: float) -> float:
    """
    Calculate P&L in currency based on lot size.

    Forex: 1 lot = 100,000 units, 0.1 lot = 10,000 units
    Crypto: 1 lot = 1 coin, 0.1 lot = 0.1 coin
    Stocks: 1 lot = 100 shares (approximation)

    For simplicity, we use: P&L (money) = P&L (points) * lot_size * 10000
    This works reasonably for forex pairs where price is around 1-150
    """
    # Check if crypto (has dash in symbol)
    if '-' in symbol and symbol.endswith('-USD'):
        # Crypto: 1 lot = 1 coin
        return pnl_points * lot_size
    elif '=' in symbol:
        # Forex: 1 lot = 100,000 units, 0.1 lot = 10,000 units
        # P&L in currency = P&L points * 10,000 * lot_size
        return pnl_points * 10000 * lot_size
    else:
        # Stocks: approximate as shares
        return pnl_points * 100 * lot_size

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(df: pd.DataFrame, atr_multiplier: float = 1.5) -> list:
    """
    Run backtest with EMA crossover strategy.
    Entry: BUY when EMA_20 crosses ABOVE EMA_50, SELL when crosses BELOW
    Exit: ATR trailing stop OR opposite crossover
    """
    trades = []
    position = None
    entry_price = 0
    entry_time = None
    atr_stop = 0

    prev_ema20 = None
    prev_ema50 = None

    for i, (idx, row) in enumerate(df.iterrows()):
        curr_ema20 = row['EMA_20']
        curr_ema50 = row['EMA_50']
        curr_close = row['Close']
        curr_high = row['High']
        curr_low = row['Low']
        curr_atr = row['ATR']

        if pd.isna(curr_ema20) or pd.isna(curr_ema50) or pd.isna(curr_atr):
            prev_ema20 = curr_ema20
            prev_ema50 = curr_ema50
            continue

        if prev_ema20 is None or prev_ema50 is None:
            prev_ema20 = curr_ema20
            prev_ema50 = curr_ema50
            continue

        # === CLOSE POSITION ===
        if position == 'buy':
            new_stop = curr_close - (atr_multiplier * curr_atr)
            if new_stop > atr_stop:
                atr_stop = new_stop

            if curr_low <= atr_stop:
                trades.append({
                    'num': len(trades) + 1,
                    'direction': 'BUY',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': idx,
                    'exit_price': atr_stop,
                    'pnl_points': atr_stop - entry_price,
                    'exit_reason': 'Stop Loss'
                })
                position = None

            elif prev_ema20 >= prev_ema50 and curr_ema20 < curr_ema50:
                trades.append({
                    'num': len(trades) + 1,
                    'direction': 'BUY',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': idx,
                    'exit_price': curr_close,
                    'pnl_points': curr_close - entry_price,
                    'exit_reason': 'Opposite Signal'
                })
                position = None

        elif position == 'sell':
            new_stop = curr_close + (atr_multiplier * curr_atr)
            if new_stop < atr_stop:
                atr_stop = new_stop

            if curr_high >= atr_stop:
                trades.append({
                    'num': len(trades) + 1,
                    'direction': 'SELL',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': idx,
                    'exit_price': atr_stop,
                    'pnl_points': entry_price - atr_stop,
                    'exit_reason': 'Stop Loss'
                })
                position = None

            elif prev_ema20 <= prev_ema50 and curr_ema20 > curr_ema50:
                trades.append({
                    'num': len(trades) + 1,
                    'direction': 'SELL',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': idx,
                    'exit_price': curr_close,
                    'pnl_points': entry_price - curr_close,
                    'exit_reason': 'Opposite Signal'
                })
                position = None

        # === NEW ENTRY ===
        if position is None:
            # BUY: EMA_20 crosses ABOVE EMA_50
            if prev_ema20 <= prev_ema50 and curr_ema20 > curr_ema50:
                position = 'buy'
                entry_price = curr_close
                entry_time = idx
                atr_stop = curr_close - (atr_multiplier * curr_atr)

            # SELL: EMA_20 crosses BELOW EMA_50
            elif prev_ema20 >= prev_ema50 and curr_ema20 < curr_ema50:
                position = 'sell'
                entry_price = curr_close
                entry_time = idx
                atr_stop = curr_close + (atr_multiplier * curr_atr)

        prev_ema20 = curr_ema20
        prev_ema50 = curr_ema50

    # Close open position
    if position is not None and len(df) > 0:
        last_close = df.iloc[-1]['Close']
        last_time = df.index[-1]

        if position == 'buy':
            trades.append({
                'num': len(trades) + 1,
                'direction': 'BUY',
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': last_time,
                'exit_price': last_close,
                'pnl_points': last_close - entry_price,
                'exit_reason': 'End of Data'
            })
        else:
            trades.append({
                'num': len(trades) + 1,
                'direction': 'SELL',
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': last_time,
                'exit_price': last_close,
                'pnl_points': entry_price - last_close,
                'exit_reason': 'End of Data'
            })

    return trades

# =============================================================================
# SUMMARY CALCULATION
# =============================================================================

def calculate_summary(trades: list, symbol: str, lot_size: float) -> dict:
    """Calculate comprehensive summary statistics."""
    if not trades:
        return {
            'total_trades': 0, 'win_rate': 0,
            'net_profit_points': 0, 'net_profit_money': 0,
            'gross_profit': 0, 'gross_loss': 0,
            'avg_profit': 0, 'avg_win': 0, 'avg_loss': 0,
            'max_drawdown': 0, 'max_consecutive_losses': 0,
            'buy_trades': 0, 'sell_trades': 0,
            'winning_trades': 0, 'losing_trades': 0,
            'largest_win': 0, 'largest_loss': 0,
            'profit_factor': 0, 'expectancy': 0
        }

    total = len(trades)
    winning = [t for t in trades if t['pnl_points'] > 0]
    losing = [t for t in trades if t['pnl_points'] <= 0]

    # P&L in points
    net_profit_points = sum(t['pnl_points'] for t in trades)
    gross_profit = sum(t['pnl_points'] for t in winning)
    gross_loss = abs(sum(t['pnl_points'] for t in losing))

    # P&L in money
    net_profit_money = calculate_pnl_money(net_profit_points, symbol, lot_size)
    gross_profit_money = calculate_pnl_money(gross_profit, symbol, lot_size) if gross_profit > 0 else 0
    gross_loss_money = calculate_pnl_money(gross_loss, symbol, lot_size)

    # Max drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cumulative += t['pnl_points']
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for t in trades:
        if t['pnl_points'] <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    # Stats
    largest_win = max([t['pnl_points'] for t in winning]) if winning else 0
    largest_loss = min([t['pnl_points'] for t in losing]) if losing else 0
    avg_win = gross_profit / len(winning) if winning else 0
    avg_loss = gross_loss / len(losing) if losing else 0

    # Profit factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

    # Expectancy
    expectancy = net_profit_points / total if total > 0 else 0

    return {
        'total_trades': total,
        'win_rate': round((len(winning) / total) * 100, 2) if total > 0 else 0,
        'net_profit_points': round(net_profit_points, 5),
        'net_profit_money': round(net_profit_money, 2),
        'gross_profit': round(gross_profit, 5),
        'gross_loss': round(gross_loss, 5),
        'avg_profit': round(net_profit_points / total, 5) if total > 0 else 0,
        'avg_win': round(avg_win, 5),
        'avg_loss': round(avg_loss, 5),
        'max_drawdown': round(max_dd, 5),
        'max_consecutive_losses': max_consec,
        'buy_trades': len([t for t in trades if t['direction'] == 'BUY']),
        'sell_trades': len([t for t in trades if t['direction'] == 'SELL']),
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'largest_win': round(largest_win, 5),
        'largest_loss': round(largest_loss, 5),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else '∞',
        'expectancy': round(expectancy, 5)
    }

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.title("📈 Backtesting Tool")
st.caption("EMA Crossover + ATR Trailing Stop Strategy")

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("⚙️ Settings")

    # Symbol selection with "Other" option
    st.subheader("Symbol")

    symbol_options = ['Select...'] + SORTED_SYMBOLS + ['Other (Custom)']
    selected_display = st.selectbox("Asset", options=symbol_options, index=0)

    symbol = None
    custom_symbol = None

    if selected_display == 'Other (Custom)':
        custom_symbol = st.text_input(
            "Enter Symbol",
            value="",
            placeholder="e.g., EURUSD=X, BTC-USD",
            help="Enter Yahoo Finance symbol"
        )
        if custom_symbol:
            symbol = custom_symbol
            st.caption(f"Using: {symbol}")
    elif selected_display != 'Select...':
        symbol = ALL_SYMBOLS.get(selected_display)
        st.caption(f"Symbol: {symbol}")

    st.divider()

    # Timeframe
    st.subheader("Timeframe")
    timeframe = st.selectbox(
        "Interval",
        options=['5m', '15m', '1h', '1d'],
        index=2
    )

    st.divider()

    # Period
    st.subheader("Period")
    period = st.selectbox(
        "Date Range",
        options=['1 Month', '3 Months', '6 Months', '1 Year', 'Max Available'],
        index=3
    )

    st.divider()

    # Risk Management
    st.subheader("Risk Management")

    lot_size = st.number_input(
        "Lot Size",
        min_value=0.01,
        max_value=100.0,
        value=0.1,
        step=0.01,
        help="Trading lot size (0.1 = 10,000 units for forex)"
    )

    atr_multiplier = st.slider(
        "ATR Multiplier",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1,
        help="Stop loss distance in ATR units"
    )

    st.divider()

    run = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

# =============================================================================
# MAIN CONTENT
# =============================================================================

if run:
    # Validate symbol
    if not symbol:
        st.error("❌ Please select or enter a symbol.")
        st.stop()

    # Fetch data
    with st.spinner(f"Fetching data for {symbol}..."):
        try:
            df = fetch_data(symbol, period, timeframe)
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Data error: {str(e)}")
            st.stop()

    if df is None or df.empty:
        st.error("❌ No data available.")
        st.stop()

    if len(df) < 50:
        st.warning(f"⚠️ Limited data ({len(df)} candles). Results may not be reliable.")

    # Calculate indicators
    with st.spinner("Calculating indicators..."):
        try:
            df = calculate_indicators(df)
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Indicator error: {str(e)}")
            st.stop()

    # Run backtest
    with st.spinner("Running backtest..."):
        try:
            trades = run_backtest(df, atr_multiplier)
            summary = calculate_summary(trades, symbol, lot_size)
        except Exception as e:
            st.error(f"❌ Backtest error: {str(e)}")
            st.stop()

    # Data info
    start_dt = df.index[0].strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'
    end_dt = df.index[-1].strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'
    display_name = selected_display if selected_display != 'Other (Custom)' else custom_symbol

    st.info(f"📊 {len(df)} candles | {start_dt} to {end_dt} | {display_name} ({timeframe}) | Lot Size: {lot_size}")

    st.divider()

    # =============================================================================
    # METRICS - ROW 1 (Main)
    # =============================================================================

    st.subheader("📊 Performance Summary")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Total Trades",
        summary['total_trades'],
        help="Total number of completed trades"
    )
    m2.metric(
        "Win Rate",
        f"{summary['win_rate']}%",
        delta_color="normal" if summary['win_rate'] >= 50 else "inverse"
    )
    m3.metric(
        "Net Profit ($)",
        f"${summary['net_profit_money']:.2f}",
        delta_color="normal" if summary['net_profit_money'] >= 0 else "inverse",
        help=f"Net P&L in currency (lot size: {lot_size})"
    )
    m4.metric(
        "Max Drawdown",
        f"{summary['max_drawdown']:.5f}",
        delta_color="inverse"
    )

    # =============================================================================
    # METRICS - ROW 2 (P&L Details)
    # =============================================================================

    st.subheader("💰 P&L Details")

    p1, p2, p3, p4 = st.columns(4)

    p1.metric(
        "Gross Profit",
        f"${calculate_pnl_money(summary['gross_profit'], symbol, lot_size):.2f}",
        delta_color="normal"
    )
    p2.metric(
        "Gross Loss",
        f"${calculate_pnl_money(summary['gross_loss'], symbol, lot_size):.2f}",
        delta_color="inverse"
    )
    p3.metric(
        "Profit Factor",
        f"{summary['profit_factor']}",
        help="Gross profit / Gross loss (>1 is good)"
    )
    p4.metric(
        "Expectancy",
        f"{summary['expectancy']:.5f}",
        help="Average profit per trade in points"
    )

    # =============================================================================
    # METRICS - ROW 3 (Trade Stats)
    # =============================================================================

    st.subheader("📈 Trade Statistics")

    t1, t2, t3, t4 = st.columns(4)

    t1.metric("Buy Trades", summary['buy_trades'])
    t2.metric("Sell Trades", summary['sell_trades'])
    t3.metric("Winning Trades", summary['winning_trades'], delta_color="normal")
    t4.metric("Losing Trades", summary['losing_trades'], delta_color="inverse")

    # =============================================================================
    # METRICS - ROW 4 (Averages & Extremes)
    # =============================================================================

    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "Avg Win",
        f"{summary['avg_win']:.5f}",
        help="Average winning trade in points"
    )
    a2.metric(
        "Avg Loss",
        f"{summary['avg_loss']:.5f}",
        help="Average losing trade in points"
    )
    a3.metric(
        "Largest Win",
        f"{summary['largest_win']:.5f}",
        delta_color="normal"
    )
    a4.metric(
        "Largest Loss",
        f"{summary['largest_loss']:.5f}",
        delta_color="inverse"
    )

    # =============================================================================
    # METRICS - ROW 5 (Risk)
    # =============================================================================

    r1, r2 = st.columns(2)

    r1.metric(
        "Max Consecutive Losses",
        summary['max_consecutive_losses'],
        help="Maximum number of consecutive losing trades"
    )
    r2.metric(
        "Avg Profit/Trade",
        f"{summary['avg_profit']:.5f}",
        help="Net profit divided by total trades"
    )

    st.divider()

    # =============================================================================
    # EQUITY CURVE
    # =============================================================================

    st.subheader("📈 Equity Curve")

    if trades and len(trades) > 0:
        equity_data = []
        cumulative = 0
        for trade in trades:
            cumulative += trade['pnl_points']
            equity_data.append({
                'time': trade['exit_time'].strftime('%Y-%m-%d %H:%M') if hasattr(trade['exit_time'], 'strftime') else str(trade['exit_time']),
                'equity_points': round(cumulative, 5),
                'equity_money': round(calculate_pnl_money(cumulative, symbol, lot_size), 2)
            })

        equity_df = pd.DataFrame(equity_data)

        fig = go.Figure()

        # Equity in money
        fig.add_trace(go.Scatter(
            x=equity_df['time'],
            y=equity_df['equity_money'],
            mode='lines+markers',
            name='Equity ($)',
            line=dict(color='#58a6ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(88, 166, 255, 0.15)',
            hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>'
        ))

        fig.add_hline(y=0, line_dash="dot", line_color="#8b949e", opacity=0.5)

        fig.update_layout(
            template='plotly_dark',
            hovermode='x unified',
            height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#30363d'),
            yaxis=dict(gridcolor='#30363d', title='P&L ($)')
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades generated with the current settings.")

    st.divider()

    # =============================================================================
    # TRADE TABLE
    # =============================================================================

    st.subheader("📋 Trade History")

    if trades and len(trades) > 0:
        # Prepare display data
        display_trades = []
        for t in trades:
            pnl_money = calculate_pnl_money(t['pnl_points'], symbol, lot_size)
            entry_str = t['entry_time'].strftime('%Y-%m-%d %H:%M') if hasattr(t['entry_time'], 'strftime') else str(t['entry_time'])
            exit_str = t['exit_time'].strftime('%Y-%m-%d %H:%M') if hasattr(t['exit_time'], 'strftime') else str(t['exit_time'])

            display_trades.append({
                '#': t['num'],
                'Direction': t['direction'],
                'Entry Time': entry_str,
                'Entry Price': round(t['entry_price'], 5),
                'Exit Time': exit_str,
                'Exit Price': round(t['exit_price'], 5),
                'P&L (pts)': round(t['pnl_points'], 5),
                'P&L ($)': round(pnl_money, 2),
                'P&L %': round((t['pnl_points'] / t['entry_price']) * 100, 2),
                'Exit Reason': t['exit_reason']
            })

        trades_df = pd.DataFrame(display_trades)

        # HTML table with colors
        html = '<table style="width:100%; border-collapse:collapse; font-size:0.85rem;">'
        html += '<thead><tr style="background:#161b22; color:#8b949e;">'
        headers = ['#', 'Dir', 'Entry Time', 'Entry', 'Exit Time', 'Exit', 'P&L (pts)', 'P&L ($)', 'P&L %', 'Exit']
        for h in headers:
            html += f'<th style="padding:10px; text-align:left; border-bottom:1px solid #30363d;">{h}</th>'
        html += '</tr></thead><tbody>'

        for _, row in trades_df.iterrows():
            pnl_class = 'color:#3fb950;' if row['P&L (pts)'] >= 0 else 'color:#f85149;'
            dir_class = 'color:#3fb950; font-weight:600;' if row['Direction'] == 'BUY' else 'color:#f85149; font-weight:600;'
            pnl_sign = '+' if row['P&L (pts)'] >= 0 else ''

            html += '<tr style="color:#c9d1d9;">'
            html += f"<td style='padding:8px; border-bottom:1px solid #30363d;'>{row['#']}</td>"
            html += f"<td style='padding:8px; border-bottom:1px solid #30363d; {dir_class}'>{row['Direction']}</td>"
            html += f"<td style='padding:8px; border-bottom:1px solid #30363d;'>{row['Entry Time']}</td>"
            html += f"<td style='padding:8px; border-bottom:1px solid #30363d;'>{row['Entry Price']}</td>"
            html += f"<td style='padding:8px; border-bottom:1px solid #30363d;'>{row['Exit Time']}</td>"
            html += f"<td style='padding:8px; border-bottom:1px solid #30363d;'>{row['Exit Price']}</td>"
            html += f"<td style='padding:8px; border-bottom:1px solid #30363d; {pnl_class}'>{pnl_sign}{row['P&L (pts)']}</td>"
            html += f"<td style='padding:8px; border-bottom:1px solid #30363d; {pnl_class}'>{pnl_sign}${row['P&L ($)']:.2f}</td>"
            html += f"<td style='padding:8px; border-bottom:1px solid #30363d; {pnl_class}'>{pnl_sign}{row['P&L %']:.2f}%</td>"
            html += f"<td style='padding:8px; border-bottom:1px solid #30363d;'>{row['Exit Reason']}</td>"
            html += '</tr>'

        html += '</tbody></table>'

        st.markdown(html, unsafe_allow_html=True)

        # CSV Download
        csv_data = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade History (CSV)",
            data=csv_data,
            file_name=f"backtest_{display_name.replace('/', '_')}_{timeframe}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ No trades found. Try a different period or timeframe.")

else:
    # Welcome screen
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Strategy")
        st.markdown("""
        **EMA Crossover + ATR Trailing Stop**

        **Entry Rules:**
        - **BUY**: EMA 20 crosses above EMA 50
        - **SELL**: EMA 20 crosses below EMA 50

        **Exit Rules:**
        - ATR trailing stop (moves only in profit direction)
        - Stop loss hit OR opposite crossover

        **P&L Calculation:**
        - P&L in points based on price movement
        - P&L in $ based on lot size (0.1 lot = 10,000 units)
        """)

    with col2:
        st.subheader("📌 Available Symbols")
        st.markdown("""
        **Forex (24 pairs):**
        EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF...

        **Crypto (12 coins):**
        BTC/USD, ETH/USD, SOL/USD, BNB/USD...

        **Stocks (12 assets):**
        AAPL, TSLA, MSFT, GOOGL, NVDA...

        **Or select "Other" to enter any Yahoo Finance symbol**
        """)

    st.divider()
    st.info("👈 Configure settings in the sidebar and click **Run Backtest**.")
