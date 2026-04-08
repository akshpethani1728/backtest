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

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Backtesting Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3, h4 { color: #f0f6fc !important; }
    .stMarkdown, .stText { color: #c9d1d9; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; }
    [data-testid="stMetricValue"] { color: #f0f6fc !important; font-weight: 600 !important; }
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    hr {border-color: #30363d; margin: 1rem 0;}
    .section-header {
        background: linear-gradient(90deg, #161b22 0%, transparent 100%);
        padding: 10px 15px;
        border-left: 3px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        margin-bottom: 15px;
    }
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
    },
    'Crypto': {
        'BTC/USD': 'BTC-USD', 'ETH/USD': 'ETH-USD', 'SOL/USD': 'SOL-USD',
        'BNB/USD': 'BNB-USD', 'XRP/USD': 'XRP-USD', 'ADA/USD': 'ADA-USD',
        'DOGE/USD': 'DOGE-USD', 'DOT/USD': 'DOT-USD', 'AVAX/USD': 'AVAX-USD',
        'LINK/USD': 'LINK-USD', 'MATIC/USD': 'MATIC-USD', 'LTC/USD': 'LTC-USD',
    },
    'Stocks': {
        'Apple': 'AAPL', 'Tesla': 'TSLA', 'Microsoft': 'MSFT',
        'Google': 'GOOGL', 'Amazon': 'AMZN', 'NVIDIA': 'NVDA',
        'Meta': 'META', 'Netflix': 'NFLX', 'AMD': 'AMD',
        'Gold': 'GC=F', 'Oil': 'CL=F',
    }
}

ALL_SYMBOLS = {}
for category, pairs in SYMBOLS.items():
    for name, symbol in pairs.items():
        ALL_SYMBOLS[name] = symbol

# =============================================================================
# SESSION STATE
# =============================================================================

def init_session_state():
    defaults = {
        'last_symbol': 'EUR/USD',
        'last_timeframe': '1h',
        'last_period': '1 Year',
        'last_lot': 0.1,
        'last_atr': 1.5,
        'recent_symbols': [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def save_to_recent(symbol_name):
    if symbol_name not in st.session_state.recent_symbols:
        st.session_state.recent_symbols.insert(0, symbol_name)
        st.session_state.recent_symbols = st.session_state.recent_symbols[:5]

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
    if '-' in symbol and symbol.endswith('-USD'):
        return entry_price * lot_size
    elif '=' in symbol:
        return entry_price * 10000 * lot_size
    else:
        return entry_price * 100 * lot_size

def calculate_roi(pnl_money: float, investment: float) -> float:
    if investment > 0:
        return (pnl_money / investment) * 100
    return 0

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(df: pd.DataFrame, atr_multiplier: float = 1.5) -> list:
    """
    Enhanced backtest with ATR trend-based exits.
    Exit: ATR Stop, EMA Cross, or ATR Reversal
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

        atr_expanding = prev_atr is not None and curr_atr > prev_atr

        # === CLOSE POSITION ===
        if position == 'buy':
            new_stop = curr_close - (atr_multiplier * curr_atr)
            if new_stop > atr_stop:
                atr_stop = new_stop

            if curr_low <= atr_stop:
                trades.append({
                    'num': len(trades) + 1, 'direction': 'BUY',
                    'entry_time': entry_time, 'entry_price': entry_price,
                    'exit_time': idx, 'exit_price': atr_stop,
                    'pnl_points': atr_stop - entry_price, 'exit_reason': 'ATR Stop'
                })
                position = None
            elif prev_ema20 >= prev_ema50 and curr_ema20 < curr_ema50:
                trades.append({
                    'num': len(trades) + 1, 'direction': 'BUY',
                    'entry_time': entry_time, 'entry_price': entry_price,
                    'exit_time': idx, 'exit_price': curr_close,
                    'pnl_points': curr_close - entry_price, 'exit_reason': 'EMA Cross'
                })
                position = None
            elif atr_expanding and curr_atr > entry_atr * 1.2 and curr_close < entry_price:
                trades.append({
                    'num': len(trades) + 1, 'direction': 'BUY',
                    'entry_time': entry_time, 'entry_price': entry_price,
                    'exit_time': idx, 'exit_price': curr_close,
                    'pnl_points': curr_close - entry_price, 'exit_reason': 'ATR Reversal'
                })
                position = None

        elif position == 'sell':
            new_stop = curr_close + (atr_multiplier * curr_atr)
            if new_stop < atr_stop:
                atr_stop = new_stop

            if curr_high >= atr_stop:
                trades.append({
                    'num': len(trades) + 1, 'direction': 'SELL',
                    'entry_time': entry_time, 'entry_price': entry_price,
                    'exit_time': idx, 'exit_price': atr_stop,
                    'pnl_points': entry_price - atr_stop, 'exit_reason': 'ATR Stop'
                })
                position = None
            elif prev_ema20 <= prev_ema50 and curr_ema20 > curr_ema50:
                trades.append({
                    'num': len(trades) + 1, 'direction': 'SELL',
                    'entry_time': entry_time, 'entry_price': entry_price,
                    'exit_time': idx, 'exit_price': curr_close,
                    'pnl_points': entry_price - curr_close, 'exit_reason': 'EMA Cross'
                })
                position = None
            elif atr_expanding and curr_atr > entry_atr * 1.2 and curr_close > entry_price:
                trades.append({
                    'num': len(trades) + 1, 'direction': 'SELL',
                    'entry_time': entry_time, 'entry_price': entry_price,
                    'exit_time': idx, 'exit_price': curr_close,
                    'pnl_points': entry_price - curr_close, 'exit_reason': 'ATR Reversal'
                })
                position = None

        # === NEW ENTRY ===
        if position is None:
            if prev_ema20 <= prev_ema50 and curr_ema20 > curr_ema50:
                position = 'buy'
                entry_price = curr_close
                entry_time = idx
                entry_atr = curr_atr
                atr_stop = curr_close - (atr_multiplier * curr_atr)
            elif prev_ema20 >= prev_ema50 and curr_ema20 < curr_ema50:
                position = 'sell'
                entry_price = curr_close
                entry_time = idx
                entry_atr = curr_atr
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
            'exit_time': last_time, 'exit_price': last_close,
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

    investments = [calculate_investment(symbol, t['entry_price'], lot_size) for t in trades]
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
# STREAMLIT UI
# =============================================================================

st.title("📈 Professional Backtesting Tool")
st.caption("EMA Crossover Strategy with ATR Trailing Stop")

# =============================================================================
# SIDEBAR - SIMPLE & STRUCTURED
# =============================================================================

with st.sidebar:
    st.markdown("### 📊 Select Symbol")

    # Symbol dropdown
    symbol_options = list(ALL_SYMBOLS.keys())
    try:
        default_idx = symbol_options.index(st.session_state.last_symbol)
    except ValueError:
        default_idx = 0

    selected_symbol = st.selectbox(
        "Choose Asset",
        options=symbol_options,
        index=default_idx,
        help="Select from forex, crypto, or stocks"
    )
    st.session_state.last_symbol = selected_symbol

    st.divider()

    # Settings in clean sections
    st.markdown("### ⚙️ Time Settings")

    timeframe = st.selectbox(
        "Timeframe",
        options=['5m', '15m', '1h', '4h', '1d'],
        index=2
    )

    period = st.selectbox(
        "Period",
        options=['1 Month', '3 Months', '6 Months', '1 Year', 'Max Available'],
        index=3
    )

    st.divider()

    st.markdown("### 💰 Trade Settings")

    lot_size = st.number_input(
        "Lot Size",
        min_value=0.01,
        max_value=100.0,
        value=0.1,
        step=0.01
    )

    atr_multiplier = st.selectbox(
        "ATR Multiplier",
        options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
        index=2
    )

    st.divider()

    run = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

# =============================================================================
# MAIN CONTENT
# =============================================================================

if run:
    symbol = ALL_SYMBOLS.get(selected_symbol, selected_symbol)

    # Save settings
    st.session_state.last_symbol = selected_symbol
    st.session_state.last_timeframe = timeframe
    st.session_state.last_period = period
    st.session_state.last_lot = lot_size
    st.session_state.last_atr = atr_multiplier

    save_to_recent(selected_symbol)

    # Fetch data
    with st.spinner("📥 Fetching data..."):
        try:
            df = fetch_data(symbol, period, timeframe)
        except Exception as e:
            st.error(f"❌ {str(e)}")
            st.stop()

    if len(df) < 50:
        st.warning(f"⚠️ Limited data ({len(df)} candles)")

    # Calculate indicators
    with st.spinner("🔢 Calculating..."):
        try:
            df = calculate_indicators(df)
        except Exception as e:
            st.error(f"❌ {str(e)}")
            st.stop()

    with st.spinner("📊 Running backtest..."):
        try:
            trades = run_backtest(df, atr_multiplier)
            summary = calculate_summary(trades, symbol, lot_size)
        except Exception as e:
            st.error(f"❌ {str(e)}")
            st.stop()

    # Info bar
    start_dt = df.index[0].strftime('%Y-%m-%d')
    end_dt = df.index[-1].strftime('%Y-%m-%d')
    st.success(f"✅ {len(df)} candles | {start_dt} → {end_dt} | {selected_symbol} | Lot: {lot_size} | ATR×{atr_multiplier}")

    st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "📈 Equity Curve", "📋 Trades"])

    with tab1:
        # Key Metrics
        st.markdown("### Key Metrics")
        c1, c2, c3, c4 = st.columns(4)

        delta_color = "normal" if summary['net_profit_money'] >= 0 else "inverse"
        c1.metric("Total Trades", summary['total_trades'])
        c2.metric("Win Rate", f"{summary['win_rate']}%",
                  delta_color="normal" if summary['win_rate'] >= 50 else "inverse")
        c3.metric("Net P&L", f"${summary['net_profit_money']:.2f}", delta_color=delta_color)
        c4.metric("Max DD", f"{summary['max_drawdown']:.5f}", delta_color="inverse")

        st.divider()

        # Investment & ROI
        st.markdown("### 💵 Investment & Returns")
        inv1, inv2, inv3, inv4 = st.columns(4)

        roi_color = "normal" if summary['total_roi'] >= 0 else "inverse"
        inv1.metric("Total Investment", f"${summary['total_investment']:.2f}")
        inv2.metric("Net Profit", f"${summary['net_profit_money']:.2f}", delta_color=roi_color)
        inv3.metric("Total ROI", f"{summary['total_roi']:.2f}%", delta_color=roi_color)
        inv4.metric("Avg ROI/Trade", f"{summary['avg_roi']:.2f}%", delta_color=roi_color)

        st.divider()

        # Profit & Loss
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

        # Trade Breakdown
        st.markdown("### 📈 Trade Breakdown")
        t1, t2, t3, t4, t5 = st.columns(5)

        t1.metric("Buy", summary['buy_trades'])
        t2.metric("Sell", summary['sell_trades'])
        t3.metric("Win", summary['winning_trades'], delta_color="normal")
        t4.metric("Loss", summary['losing_trades'], delta_color="inverse")
        t5.metric("Consec Loss", summary['max_consecutive_losses'], delta_color="inverse")

        st.divider()

        # Averages
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
                    'Entry Time': t['entry_time'].strftime('%m/%d %H:%M') if hasattr(t['entry_time'], 'strftime') else str(t['entry_time']),
                    'Entry Price': round(t['entry_price'], 5),
                    'Investment': round(investment, 2),
                    'Exit Time': t['exit_time'].strftime('%m/%d %H:%M') if hasattr(t['exit_time'], 'strftime') else str(t['exit_time']),
                    'Exit Price': round(t['exit_price'], 5),
                    'P&L (pts)': round(t['pnl_points'], 5),
                    'P&L ($)': round(pnl_money, 2),
                    'ROI %': round(roi, 2),
                    'Exit Reason': t['exit_reason']
                })

            trades_df = pd.DataFrame(display_trades)

            # Simple HTML table
            html = '<table class="trade-table">'
            html += '<thead><tr>'
            headers = ['#', 'Dir', 'Entry Time', 'Entry Price', 'Inv ($)', 'Exit Time', 'Exit Price', 'P&L (pts)', 'P&L ($)', 'ROI %', 'Exit']
            for h in headers:
                html += f'<th>{h}</th>'
            html += '</tr></thead><tbody>'

            for _, row in trades_df.iterrows():
                pnl_val = row['P&L (pts)']
                pnl_color = '#3fb950' if pnl_val >= 0 else '#f85149'
                dir_color = '#3fb950' if row['Direction'] == 'BUY' else '#f85149'
                sign = '+' if pnl_val >= 0 else ''

                html += '<tr>'
                html += f"<td>{row['#']}</td>"
                html += f"<td style='color:{dir_color};font-weight:600;'>{row['Direction']}</td>"
                html += f"<td>{row['Entry Time']}</td>"
                html += f"<td>{row['Entry Price']}</td>"
                html += f"<td>${row['Investment']:.2f}</td>"
                html += f"<td>{row['Exit Time']}</td>"
                html += f"<td>{row['Exit Price']}</td>"
                html += f"<td style='color:{pnl_color};'>{sign}{pnl_val}</td>"
                html += f"<td style='color:{pnl_color};'>{sign}${row['P&L ($)']:.2f}</td>"
                html += f"<td style='color:{pnl_color};'>{sign}{row['ROI %']:.2f}%</td>"
                html += f"<td>{row['Exit Reason']}</td>"
                html += '</tr>'

            html += '</tbody></table>'
            st.markdown(html, unsafe_allow_html=True)

            # Download CSV
            csv = trades_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, f"backtest_{selected_symbol}_{timeframe}.csv", "text/csv")
        else:
            st.warning("⚠️ No trades found")

else:
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📈 Strategy

        **EMA Crossover + ATR Trailing Stop**

        **Entry:**
        - BUY when EMA 20 crosses above EMA 50
        - SELL when EMA 20 crosses below EMA 50

        **Exit:**
        - ATR Stop (trailing)
        - EMA Cross (opposite signal)
        - ATR Reversal (volatility spike)
        """)

    with col2:
        st.markdown("""
        ### 📌 Symbols Available

        **Forex:** 12 pairs

        **Crypto:** 12 coins

        **Stocks:** 11 assets

        Total: 35+ symbols
        """)

    st.divider()
    st.info("👈 Select a symbol from the dropdown above and click **Run Backtest**")
