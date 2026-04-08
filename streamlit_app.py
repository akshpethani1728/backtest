"""
Backtesting Application - EMA Crossover + ATR Trailing Stop
Professional Grade - Optimized for Real Trading
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Backtesting Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional Dark Theme
st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3, h4 { color: #f0f6fc !important; }
    .stMarkdown, .stText { color: #c9d1d9; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; }
    [data-testid="stMetricValue"] { color: #f0f6fc !important; font-weight: 600 !important; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        transition: all 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.2s;
    }

    /* Quick select buttons */
    .quick-btn {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 8px 12px;
        color: #c9d1d9;
        cursor: pointer;
        transition: all 0.2s;
    }
    .quick-btn:hover {
        background-color: #30363d;
        border-color: #58a6ff;
    }
    .quick-btn.selected {
        background-color: #238636;
        border-color: #238636;
        color: #ffffff;
    }

    /* Sections */
    .section-header {
        background: linear-gradient(90deg, #161b22 0%, transparent 100%);
        padding: 10px 15px;
        border-left: 3px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        margin-bottom: 15px;
    }

    /* Tabs */
    .stTabs [data-selected] {
        background-color: #238636 !important;
        color: #ffffff !important;
        border-radius: 8px 8px 0 0;
    }

    /* Tables */
    .trade-table { width: 100%; border-collapse: collapse; }
    .trade-table th {
        background: #161b22;
        padding: 12px 8px;
        text-align: left;
        border-bottom: 2px solid #30363d;
        color: #8b949e;
        font-size: 0.8rem;
        text-transform: uppercase;
    }
    .trade-table td {
        padding: 10px 8px;
        border-bottom: 1px solid #21262d;
        font-size: 0.85rem;
    }
    .trade-table tr:hover { background-color: #161b22; }

    /* Cards */
    .result-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }

    /* Hide elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {display: none;}

    /* Custom dividers */
    hr {border-color: #30363d; margin: 1rem 0;}

    /* Sidebar sections */
    .sidebar-section {
        background: #1c2128;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SYMBOLS DATABASE
# =============================================================================

SYMBOLS = {
    'Forex': {
        'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X', 'USD/JPY': 'USDJPY=X',
        'USD/CHF': 'USDCHF=X', 'AUD/USD': 'AUDUSD=X', 'USD/CAD': 'USDCAD=X',
        'NZD/USD': 'NZDUSD=X', 'EUR/GBP': 'EURGBP=X', 'EUR/JPY': 'EURJPY=X',
        'GBP/JPY': 'GBPJPY=X', 'AUD/JPY': 'AUDJPY=X', 'EUR/CHF': 'EURCHF=X',
        'GBP/CHF': 'GBPCHF=X', 'AUD/NZD': 'AUDNZD=X', 'EUR/AUD': 'EURAUD=X',
        'GBP/AUD': 'GBPAUD=X', 'NZD/JPY': 'NZDJPY=X', 'CAD/JPY': 'CADJPY=X',
        'CHF/JPY': 'CHFJPY=X', 'EUR/CAD': 'EURCAD=X', 'GBP/CAD': 'GBPCAD=X',
        'AUD/CAD': 'AUDCAD=X', 'EUR/NZD': 'EURNZD=X',
    },
    'Crypto': {
        'BTC/USD': 'BTC-USD', 'ETH/USD': 'ETH-USD', 'SOL/USD': 'SOL-USD',
        'BNB/USD': 'BNB-USD', 'XRP/USD': 'XRP-USD', 'ADA/USD': 'ADA-USD',
        'DOGE/USD': 'DOGE-USD', 'DOT/USD': 'DOT-USD', 'AVAX/USD': 'AVAX-USD',
        'LINK/USD': 'LINK-USD', 'MATIC/USD': 'MATIC-USD', 'LTC/USD': 'LTC-USD',
        'ATOM/USD': 'ATOM-USD', 'UNI/USD': 'UNI-USD', 'XLM/USD': 'XLM-USD',
        'NEAR/USD': 'NEAR-USD', 'APT/USD': 'APT-USD', 'ARB/USD': 'ARB-USD',
        'OP/USD': 'OP-USD', 'INJ/USD': 'INJ-USD', 'SUI/USD': 'SUI-USD',
    },
    'Stocks': {
        'Apple': 'AAPL', 'Tesla': 'TSLA', 'Microsoft': 'MSFT',
        'Google': 'GOOGL', 'Amazon': 'AMZN', 'NVIDIA': 'NVDA',
        'Meta': 'META', 'Netflix': 'NFLX', 'AMD': 'AMD',
        'Intel': 'INTC', 'Gold': 'GC=F', 'Oil': 'CL=F',
        'Silver': 'SI=F', 'Copper': 'HG=F', 'Natural Gas': 'NG=F',
    }
}

ALL_SYMBOLS = {}
for category, pairs in SYMBOLS.items():
    for name, symbol in pairs.items():
        ALL_SYMBOLS[name] = symbol

SORTED_SYMBOLS = sorted(ALL_SYMBOLS.keys())

# =============================================================================
# SESSION STATE - PERSIST SETTINGS
# =============================================================================

def init_session_state():
    defaults = {
        'last_symbol': 'EUR/USD',
        'last_timeframe': '1h',
        'last_period': '1 Year',
        'last_lot': 0.1,
        'last_atr': 1.5,
        'recent_symbols': [],
        'favorites': ['EUR/USD', 'GBP/USD', 'BTC/USD', 'ETH/USD'],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def save_to_recent(symbol_name):
    if symbol_name not in st.session_state.recent_symbols:
        st.session_state.recent_symbols.insert(0, symbol_name)
        st.session_state.recent_symbols = st.session_state.recent_symbols[:5]

def toggle_favorite(symbol_name):
    if symbol_name in st.session_state.favorites:
        st.session_state.favorites.remove(symbol_name)
    else:
        st.session_state.favorites.append(symbol_name)

# =============================================================================
# DATA FETCHING
# =============================================================================

@st.cache_data(ttl=300)
def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        period_map = {
            '1 Month': '1mo', '3 Months': '3mo', '6 Months': '6mo',
            '1 Year': '1y', 'Max Available': '2y'
        }
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period_map.get(period, '1y'), interval=interval)

        if df is None or df.empty:
            raise ValueError(f"No data for '{symbol}'")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        df = df[df['Close'] > 0]

        if df.empty:
            raise ValueError(f"No valid data for '{symbol}'")

        return df
    except Exception as e:
        raise ValueError(str(e))

# =============================================================================
# INDICATORS
# =============================================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
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

    if df.empty or len(df) < 20:
        raise ValueError("Insufficient data for indicators")

    return df

# =============================================================================
# P&L CALCULATION
# =============================================================================

def calculate_pnl_money(pnl_points: float, symbol: str, lot_size: float) -> float:
    if '-' in symbol and symbol.endswith('-USD'):
        return pnl_points * lot_size
    elif '=' in symbol:
        return pnl_points * 10000 * lot_size
    else:
        return pnl_points * 100 * lot_size

def calculate_investment(symbol: str, entry_price: float, lot_size: float) -> float:
    """Calculate required investment/margin for a trade."""
    if '-' in symbol and symbol.endswith('-USD'):
        # Crypto: 1 lot = 1 coin
        return entry_price * lot_size
    elif '=' in symbol:
        # Forex: 1 lot = 100,000 units, 0.1 lot = 10,000 units
        return entry_price * 10000 * lot_size
    else:
        # Stocks: approximate
        return entry_price * 100 * lot_size

def calculate_roi(pnl_money: float, investment: float) -> float:
    """Calculate ROI percentage."""
    if investment > 0:
        return (pnl_money / investment) * 100
    return 0

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(df: pd.DataFrame, atr_multiplier: float = 1.5) -> list:
    """
    Enhanced backtest with ATR trend-based exits.

    Exit conditions:
    1. ATR trailing stop hit
    2. Opposite EMA crossover
    3. ATR trend reversal (ATR expanding against trade direction = early exit)
    """
    trades = []
    position = None
    entry_price = 0
    entry_time = None
    atr_stop = 0
    entry_atr = 0
    prev_ema20 = None
    prev_ema50 = None
    prev_atr = None

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
            prev_atr = curr_atr
            continue
        if prev_ema20 is None:
            prev_ema20 = curr_ema20
            prev_ema50 = curr_ema50
            prev_atr = curr_atr
            continue

        # === ATR TREND DETECTION ===
        # ATR expanding = volatility increasing, ATR contracting = calming
        atr_expanding = prev_atr is not None and curr_atr > prev_atr
        atr_contracting = prev_atr is not None and curr_atr < prev_atr

        # === CLOSE POSITION ===
        if position == 'buy':
            new_stop = curr_close - (atr_multiplier * curr_atr)
            if new_stop > atr_stop:
                atr_stop = new_stop

            # 1. ATR trailing stop hit
            if curr_low <= atr_stop:
                trades.append({
                    'num': len(trades) + 1, 'direction': 'BUY',
                    'entry_time': entry_time, 'entry_price': entry_price,
                    'entry_atr': entry_atr, 'exit_time': idx, 'exit_price': atr_stop,
                    'pnl_points': atr_stop - entry_price, 'exit_reason': 'ATR Stop'
                })
                position = None
            # 2. Opposite EMA crossover
            elif prev_ema20 >= prev_ema50 and curr_ema20 < curr_ema50:
                trades.append({
                    'num': len(trades) + 1, 'direction': 'BUY',
                    'entry_time': entry_time, 'entry_price': entry_price,
                    'entry_atr': entry_atr, 'exit_time': idx, 'exit_price': curr_close,
                    'pnl_points': curr_close - entry_price, 'exit_reason': 'EMA Cross'
                })
                position = None
            # 3. ATR EXPANDING AGAINST BUY (volatility spike against position)
            elif atr_expanding and curr_atr > entry_atr * 1.2 and curr_close < entry_price:
                # ATR expanded 20%+ AND price is below entry = adverse move
                trades.append({
                    'num': len(trades) + 1, 'direction': 'BUY',
                    'entry_time': entry_time, 'entry_price': entry_price,
                    'entry_atr': entry_atr, 'exit_time': idx, 'exit_price': curr_close,
                    'pnl_points': curr_close - entry_price, 'exit_reason': 'ATR Reversal'
                })
                position = None

        elif position == 'sell':
            new_stop = curr_close + (atr_multiplier * curr_atr)
            if new_stop < atr_stop:
                atr_stop = new_stop

            # 1. ATR trailing stop hit
            if curr_high >= atr_stop:
                trades.append({
                    'num': len(trades) + 1, 'direction': 'SELL',
                    'entry_time': entry_time, 'entry_price': entry_price,
                    'entry_atr': entry_atr, 'exit_time': idx, 'exit_price': atr_stop,
                    'pnl_points': entry_price - atr_stop, 'exit_reason': 'ATR Stop'
                })
                position = None
            # 2. Opposite EMA crossover
            elif prev_ema20 <= prev_ema50 and curr_ema20 > curr_ema50:
                trades.append({
                    'num': len(trades) + 1, 'direction': 'SELL',
                    'entry_time': entry_time, 'entry_price': entry_price,
                    'entry_atr': entry_atr, 'exit_time': idx, 'exit_price': curr_close,
                    'pnl_points': entry_price - curr_close, 'exit_reason': 'EMA Cross'
                })
                position = None
            # 3. ATR EXPANDING AGAINST SELL (volatility spike against position)
            elif atr_expanding and curr_atr > entry_atr * 1.2 and curr_close > entry_price:
                # ATR expanded 20%+ AND price is above entry = adverse move
                trades.append({
                    'num': len(trades) + 1, 'direction': 'SELL',
                    'entry_time': entry_time, 'entry_price': entry_price,
                    'entry_atr': entry_atr, 'exit_time': idx, 'exit_price': curr_close,
                    'pnl_points': entry_price - curr_close, 'exit_reason': 'ATR Reversal'
                })
                position = None

        # === NEW ENTRY ===
        if position is None:
            if prev_ema20 <= prev_ema50 and curr_ema20 > curr_ema50:
                position = 'buy'
                entry_price = curr_close
                entry_time = idx
                entry_atr = curr_atr  # Store entry ATR
                atr_stop = curr_close - (atr_multiplier * curr_atr)
            elif prev_ema20 >= prev_ema50 and curr_ema20 < curr_ema50:
                position = 'sell'
                entry_price = curr_close
                entry_time = idx
                entry_atr = curr_atr  # Store entry ATR
                atr_stop = curr_close + (atr_multiplier * curr_atr)

        prev_ema20 = curr_ema20
        prev_ema50 = curr_ema50
        prev_atr = curr_atr

    # Close at end
    if position is not None and len(df) > 0:
        last_close = df.iloc[-1]['Close']
        last_time = df.index[-1]
        direction = 'BUY' if position == 'buy' else 'SELL'
        pnl = last_close - entry_price if position == 'buy' else entry_price - last_close
        trades.append({
            'num': len(trades) + 1, 'direction': direction,
            'entry_time': entry_time, 'entry_price': entry_price,
            'entry_atr': entry_atr, 'exit_time': last_time, 'exit_price': last_close,
            'pnl_points': pnl, 'exit_reason': 'End of Data'
        })

    return trades

# =============================================================================
# SUMMARY CALCULATION
# =============================================================================

def calculate_summary(trades: list, symbol: str, lot_size: float) -> dict:
    if not trades:
        return {
            'total_trades': 0, 'win_rate': 0, 'net_profit_points': 0,
            'net_profit_money': 0, 'gross_profit': 0, 'gross_loss': 0,
            'avg_profit': 0, 'avg_win': 0, 'avg_loss': 0, 'max_drawdown': 0,
            'max_consecutive_losses': 0, 'buy_trades': 0, 'sell_trades': 0,
            'winning_trades': 0, 'losing_trades': 0, 'largest_win': 0,
            'largest_loss': 0, 'profit_factor': 0, 'expectancy': 0,
            'total_investment': 0, 'total_roi': 0, 'avg_investment': 0, 'avg_roi': 0
        }

    total = len(trades)
    winning = [t for t in trades if t['pnl_points'] > 0]
    losing = [t for t in trades if t['pnl_points'] <= 0]

    net_profit_points = sum(t['pnl_points'] for t in trades)
    gross_profit = sum(t['pnl_points'] for t in winning)
    gross_loss = abs(sum(t['pnl_points'] for t in losing))
    net_profit_money = calculate_pnl_money(net_profit_points, symbol, lot_size)

    # Calculate investment for each trade
    investments = []
    for t in trades:
        inv = calculate_investment(symbol, t['entry_price'], lot_size)
        investments.append(inv)

    total_investment = sum(investments)
    total_roi = calculate_roi(net_profit_money, total_investment) if total_investment > 0 else 0
    avg_investment = total_investment / total if total > 0 else 0
    avg_roi = (net_profit_money / avg_investment * 100) if avg_investment > 0 else 0

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

    max_consec = 0
    current_consec = 0
    for t in trades:
        if t['pnl_points'] <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    largest_win = max([t['pnl_points'] for t in winning]) if winning else 0
    largest_loss = min([t['pnl_points'] for t in losing]) if losing else 0
    avg_win = gross_profit / len(winning) if winning else 0
    avg_loss = gross_loss / len(losing) if losing else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
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
        'expectancy': round(expectancy, 5),
        'total_investment': round(total_investment, 2),
        'total_roi': round(total_roi, 2),
        'avg_investment': round(avg_investment, 2),
        'avg_roi': round(avg_roi, 2)
    }

# =============================================================================
# STREAMLIT UI - MAIN
# =============================================================================

st.title("📈 Professional Backtesting Tool")
st.caption("EMA Crossover Strategy with ATR Trailing Stop")

# =============================================================================
# SIDEBAR - ORGANIZED SECTIONS
# =============================================================================

with st.sidebar:
    st.markdown("### 🎯 Quick Select")

    # Favorite symbols as quick buttons
    st.markdown("**⭐ Favorites**")
    fav_cols = st.columns(2)
    for i, fav in enumerate(st.session_state.favorites[:4]):
        if fav_cols[i % 2].button(fav.split('/')[0], use_container_width=True):
            st.session_state.last_symbol = fav

    # Recent symbols
    if st.session_state.recent_symbols:
        st.markdown("**📌 Recent**")
        recent_cols = st.columns(2)
        for i, recent in enumerate(st.session_state.recent_symbols[:4]):
            if recent_cols[i % 2].button(recent.split('/')[0], use_container_width=True):
                st.session_state.last_symbol = recent

    st.divider()

    # Symbol Selection
    st.markdown("### 📊 Symbol")

    # Grouped by category with expander
    for category, pairs in SYMBOLS.items():
        with st.expander(f"{category} ({len(pairs)})"):
            for name, symbol in sorted(pairs.items()):
                is_fav = "⭐" if name in st.session_state.favorites else ""
                if st.button(f"{is_fav} {name}", use_container_width=True):
                    st.session_state.last_symbol = name
                    save_to_recent(name)

    st.divider()

    # Technical Settings
    st.markdown("### ⚙️ Technical Settings")

    timeframe = st.selectbox(
        "Timeframe",
        ['5m', '15m', '1h', '4h', '1d'],
        index=['5m', '15m', '1h', '4h', '1d'].index(st.session_state.last_timeframe)
    )

    period = st.selectbox(
        "Period",
        ['1 Month', '3 Months', '6 Months', '1 Year', 'Max Available'],
        index=['1 Month', '3 Months', '6 Months', '1 Year', 'Max Available'].index(st.session_state.last_period)
    )

    st.divider()

    # Risk Management
    st.markdown("### 💰 Risk Management")

    lot_size = st.number_input(
        "Lot Size",
        min_value=0.01, max_value=100.0,
        value=st.session_state.last_lot,
        step=0.01
    )

    atr_multiplier = st.selectbox(
        "ATR Multiplier",
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
        index=2
    )

    st.divider()

    run = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

if run:
    # Get symbol
    symbol_name = st.session_state.last_symbol
    symbol = ALL_SYMBOLS.get(symbol_name, symbol_name)

    # Update session state
    st.session_state.last_timeframe = timeframe
    st.session_state.last_period = period
    st.session_state.last_lot = lot_size
    st.session_state.last_atr = atr_multiplier

    save_to_recent(symbol_name)

    # Fetch data
    with st.spinner("📥 Fetching data..."):
        try:
            df = fetch_data(symbol, period, timeframe)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()

    if len(df) < 50:
        st.warning(f"⚠️ Limited data ({len(df)} candles)")

    # Calculate
    with st.spinner("🔢 Calculating indicators..."):
        try:
            df = calculate_indicators(df)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()

    with st.spinner("📊 Running backtest..."):
        trades = run_backtest(df, atr_multiplier)
        summary = calculate_summary(trades, symbol, lot_size)

    # Info bar
    start_dt = df.index[0].strftime('%Y-%m-%d')
    end_dt = df.index[-1].strftime('%Y-%m-%d')
    st.success(f"✅ {len(df)} candles | {start_dt} → {end_dt} | {symbol_name} | Lot: {lot_size} | ATR×{atr_multiplier}")

    st.divider()

    # =============================================================================
    # TABS FOR RESULTS
    # =============================================================================

    tab1, tab2, tab3 = st.tabs(["📊 Summary", "📈 Equity Curve", "📋 Trades"])

    with tab1:
        # Row 1 - Main metrics
        st.markdown("### Key Metrics")
        c1, c2, c3, c4 = st.columns(4)

        delta_color = "normal" if summary['net_profit_money'] >= 0 else "inverse"
        c1.metric("Total Trades", summary['total_trades'])
        c2.metric("Win Rate", f"{summary['win_rate']}%",
                  delta_color="normal" if summary['win_rate'] >= 50 else "inverse")
        c3.metric("Net P&L", f"${summary['net_profit_money']:.2f}", delta_color=delta_color)
        c4.metric("Max DD", f"{summary['max_drawdown']:.5f}", delta_color="inverse")

        st.divider()

        # Row 2 - Investment & ROI
        st.markdown("### 💵 Investment & Returns")
        inv1, inv2, inv3, inv4 = st.columns(4)

        roi_color = "normal" if summary['total_roi'] >= 0 else "inverse"
        inv1.metric("Total Investment", f"${summary['total_investment']:.2f}",
                   help="Total capital required for all trades")
        inv2.metric("Net Profit", f"${summary['net_profit_money']:.2f}", delta_color=roi_color,
                   help="Net profit/loss in currency")
        inv3.metric("Total ROI", f"{summary['total_roi']:.2f}%", delta_color=roi_color,
                   help="Return on Investment percentage")
        inv4.metric("Avg ROI/Trade", f"{summary['avg_roi']:.2f}%", delta_color=roi_color,
                   help="Average ROI per trade")

        st.divider()

        # Row 3 - P&L Details
        st.markdown("### 💰 Profit & Loss")
        p1, p2, p3, p4 = st.columns(4)

        pf = summary['profit_factor']
        pf_color = "normal" if (isinstance(pf, (int, float)) and pf > 1) else "inverse"

        p1.metric("Gross Profit", f"${calculate_pnl_money(summary['gross_profit'], symbol, lot_size):.2f}",
                  delta_color="normal")
        p2.metric("Gross Loss", f"${calculate_pnl_money(summary['gross_loss'], symbol, lot_size):.2f}",
                  delta_color="inverse")
        p3.metric("Profit Factor", str(pf), delta_color=pf_color)
        p4.metric("Expectancy", f"{summary['expectancy']:.5f}")

        st.divider()

        # Row 4 - Trade breakdown
        st.markdown("### 📈 Trade Breakdown")
        t1, t2, t3, t4, t5 = st.columns(5)

        t1.metric("Buy", summary['buy_trades'])
        t2.metric("Sell", summary['sell_trades'])
        t3.metric("Win", summary['winning_trades'], delta_color="normal")
        t4.metric("Loss", summary['losing_trades'], delta_color="inverse")
        t5.metric("Consec Loss", summary['max_consecutive_losses'], delta_color="inverse")

        st.divider()

        # Row 5 - Averages
        st.markdown("### 📉 Averages & Extremes")
        a1, a2, a3, a4 = st.columns(4)

        a1.metric("Avg Win", f"{summary['avg_win']:.5f}")
        a2.metric("Avg Loss", f"{summary['avg_loss']:.5f}")
        a3.metric("Best Trade", f"{summary['largest_win']:.5f}", delta_color="normal")
        a4.metric("Worst Trade", f"{summary['largest_loss']:.5f}", delta_color="inverse")

    with tab2:
        if trades:
            equity_data = []
            cumulative = 0
            for t in trades:
                cumulative += t['pnl_points']
                equity_data.append({
                    'time': t['exit_time'].strftime('%Y-%m-%d') if hasattr(t['exit_time'], 'strftime') else str(t['exit_time']),
                    'equity_points': round(cumulative, 5),
                    'equity_money': round(calculate_pnl_money(cumulative, symbol, lot_size), 2)
                })

            eq_df = pd.DataFrame(equity_data)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_df['time'], y=eq_df['equity_money'],
                mode='lines+markers',
                line=dict(color='#58a6ff', width=2),
                fill='tozeroy',
                fillcolor='rgba(88, 166, 255, 0.15)',
                hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>'
            ))
            fig.add_hline(y=0, line_dash="dot", line_color="#8b949e", opacity=0.5)

            fig.update_layout(
                template='plotly_dark',
                hovermode='x unified',
                height=450,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='#30363d'),
                yaxis=dict(gridcolor='#30363d', title='P&L ($)')
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades to display")

    with tab3:
        if trades:
            display_trades = []
            for t in trades:
                pnl_money = calculate_pnl_money(t['pnl_points'], symbol, lot_size)
                investment = calculate_investment(symbol, t['entry_price'], lot_size)
                roi = calculate_roi(pnl_money, investment)
                display_trades.append({
                    '#': t['num'],
                    'Direction': t['direction'],
                    'Entry': t['entry_time'].strftime('%m/%d %H:%M') if hasattr(t['entry_time'], 'strftime') else str(t['entry_time']),
                    'Entry Price': round(t['entry_price'], 5),
                    'Investment': f"${investment:.2f}",
                    'Exit': t['exit_time'].strftime('%m/%d %H:%M') if hasattr(t['exit_time'], 'strftime') else str(t['exit_time']),
                    'Exit Price': round(t['exit_price'], 5),
                    'P&L (pts)': round(t['pnl_points'], 5),
                    'P&L ($)': round(pnl_money, 2),
                    'ROI %': f"{roi:.2f}%",
                    'Exit Reason': t['exit_reason']
                })

            trades_df = pd.DataFrame(display_trades)

            # HTML Table
            html = '<table class="trade-table">'
            html += '<thead><tr>'
            for h in ['#', 'Dir', 'Entry', 'Entry Price', 'Inv ($)', 'Exit', 'Exit Price', 'P&L (pts)', 'P&L ($)', 'ROI %', 'Exit']:
                html += f'<th>{h}</th>'
            html += '</tr></thead><tbody>'

            for _, row in trades_df.iterrows():
                pnl_c = '#3fb950' if '+$' in str(row['P&L ($)']) or (str(row['P&L ($)']).replace('$','').replace('.','').replace('-','').isdigit() and float(str(row['P&L ($)']).replace('$','')) >= 0) else '#f85149'
                dir_c = '#3fb950' if row['Direction'] == 'BUY' else '#f85149'
                sign = '+' if row['P&L (pts)'] >= 0 else ''

                html += '<tr>'
                html += f"<td>{row['#']}</td>"
                html += f"<td style='color:{dir_c};font-weight:600;'>{row['Direction']}</td>"
                html += f"<td>{row['Entry']}</td>"
                html += f"<td>{row['Entry Price']}</td>"
                html += f"<td>{row['Investment']}</td>"
                html += f"<td>{row['Exit']}</td>"
                html += f"<td>{row['Exit Price']}</td>"
                html += f"<td style='color:{pnl_c};'>{sign}{row['P&L (pts)']}</td>"
                html += f"<td style='color:{pnl_c};'>{sign}${str(row['P&L ($)']).replace('$','')}</td>"
                html += f"<td style='color:{pnl_c};'>{row['ROI %']}</td>"
                html += f"<td>{row['Exit Reason']}</td>"
                html += '</tr>'

            html += '</tbody></table>'
            st.markdown(html, unsafe_allow_html=True)

            # Download
            csv = trades_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, f"backtest_{symbol_name}_{timeframe}.csv", "text/csv")
        else:
            st.warning("⚠️ No trades found")

else:
    # Welcome screen
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📈 Strategy
        **EMA Crossover + ATR Trailing Stop**

        **Entry:**
        - BUY when EMA 20 crosses above EMA 50
        - SELL when EMA 20 crosses below EMA 50

        **Exit:**
        - ATR trailing stop
        - Opposite crossover
        """)

    with col2:
        st.markdown("""
        ### 💡 Tips
        1. Select a **favorite** symbol for quick access
        2. Use **ATR Multiplier** to adjust stop distance
        3. Higher timeframe = more reliable signals
        4. Check **Equity Curve** for drawdown periods
        """)

    with col3:
        st.markdown("""
        ### 📌 Available
        **Forex:** 24 pairs

        **Crypto:** 20 coins

        **Stocks:** 15 assets

        **+ Custom symbols**
        """)

    st.divider()
    st.info("👈 Select a symbol from the sidebar or use Quick Select buttons, then click **Run Backtest**")
