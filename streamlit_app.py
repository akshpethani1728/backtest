"""
Backtesting Application - EMA Crossover + ATR Trailing Stop
Clean, stable, production-ready version
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
    .stTabs [data-selected="true"] { background-color: #238636; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    hr {border-color: #30363d;}
    .trade-buy {color: #3fb950; font-weight: 600;}
    .trade-sell {color: #f85149; font-weight: 600;}
    .pnl-positive {color: #3fb950;}
    .pnl-negative {color: #f85149;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA FETCHING
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch historical data using yfinance period parameter."""
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

        # Flatten multi-index columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure required columns exist
        required = ['Open', 'High', 'Low', 'Close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Clean data
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

    # EMAs
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # ATR (True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # Drop rows where indicators are NaN (first ~50 rows)
    df = df.dropna(subset=['EMA_20', 'EMA_50', 'ATR'])

    # Final clean
    df = df[['Open', 'High', 'Low', 'Close', 'EMA_20', 'EMA_50', 'ATR']]

    if df.empty or len(df) < 10:
        raise ValueError("Not enough data after indicator calculation. Try a longer period.")

    return df

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(df: pd.DataFrame, atr_multiplier: float = 1.5) -> list:
    """
    Run backtest with simple EMA crossover strategy.

    Entry:
    - BUY when EMA_20 crosses ABOVE EMA_50
    - SELL when EMA_20 crosses BELOW EMA_50

    Exit:
    - ATR trailing stop (moves only in profit direction)
    - OR opposite crossover
    """
    trades = []
    position = None  # None, 'buy', 'sell'
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

        # Skip if indicators not ready
        if pd.isna(curr_ema20) or pd.isna(curr_ema50) or pd.isna(curr_atr):
            prev_ema20 = curr_ema20
            prev_ema50 = curr_ema50
            continue

        # Skip first iteration
        if prev_ema20 is None or prev_ema50 is None:
            prev_ema20 = curr_ema20
            prev_ema50 = curr_ema50
            continue

        # === CLOSE EXISTING POSITION ===
        if position == 'buy':
            # Update ATR trailing stop (only moves UP)
            new_stop = curr_close - (atr_multiplier * curr_atr)
            if new_stop > atr_stop:
                atr_stop = new_stop

            # Check stop hit
            if curr_low <= atr_stop:
                trades.append({
                    'num': len(trades) + 1,
                    'direction': 'BUY',
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'entry_price': round(entry_price, 5),
                    'exit_time': idx.strftime('%Y-%m-%d %H:%M'),
                    'exit_price': round(atr_stop, 5),
                    'pnl': round(atr_stop - entry_price, 5),
                    'pnl_pct': round(((atr_stop - entry_price) / entry_price) * 100, 2),
                    'exit_reason': 'Stop Loss'
                })
                position = None

            # Check opposite crossover (sell signal)
            elif prev_ema20 >= prev_ema50 and curr_ema20 < curr_ema50:
                trades.append({
                    'num': len(trades) + 1,
                    'direction': 'BUY',
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'entry_price': round(entry_price, 5),
                    'exit_time': idx.strftime('%Y-%m-%d %H:%M'),
                    'exit_price': round(curr_close, 5),
                    'pnl': round(curr_close - entry_price, 5),
                    'pnl_pct': round(((curr_close - entry_price) / entry_price) * 100, 2),
                    'exit_reason': 'Opposite Signal'
                })
                position = None

        elif position == 'sell':
            # Update ATR trailing stop (only moves DOWN)
            new_stop = curr_close + (atr_multiplier * curr_atr)
            if new_stop < atr_stop:
                atr_stop = new_stop

            # Check stop hit
            if curr_high >= atr_stop:
                trades.append({
                    'num': len(trades) + 1,
                    'direction': 'SELL',
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'entry_price': round(entry_price, 5),
                    'exit_time': idx.strftime('%Y-%m-%d %H:%M'),
                    'exit_price': round(atr_stop, 5),
                    'pnl': round(entry_price - atr_stop, 5),
                    'pnl_pct': round(((entry_price - atr_stop) / entry_price) * 100, 2),
                    'exit_reason': 'Stop Loss'
                })
                position = None

            # Check opposite crossover (buy signal)
            elif prev_ema20 <= prev_ema50 and curr_ema20 > curr_ema50:
                trades.append({
                    'num': len(trades) + 1,
                    'direction': 'SELL',
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'entry_price': round(entry_price, 5),
                    'exit_time': idx.strftime('%Y-%m-%d %H:%M'),
                    'exit_price': round(curr_close, 5),
                    'pnl': round(entry_price - curr_close, 5),
                    'pnl_pct': round(((entry_price - curr_close) / entry_price) * 100, 2),
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

    # Close open position at end of data
    if position is not None and len(df) > 0:
        last_row = df.iloc[-1]
        last_close = last_row['Close']
        last_time = df.index[-1]

        if position == 'buy':
            trades.append({
                'num': len(trades) + 1,
                'direction': 'BUY',
                'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                'entry_price': round(entry_price, 5),
                'exit_time': last_time.strftime('%Y-%m-%d %H:%M'),
                'exit_price': round(last_close, 5),
                'pnl': round(last_close - entry_price, 5),
                'pnl_pct': round(((last_close - entry_price) / entry_price) * 100, 2),
                'exit_reason': 'End of Data'
            })
        else:
            trades.append({
                'num': len(trades) + 1,
                'direction': 'SELL',
                'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                'entry_price': round(entry_price, 5),
                'exit_time': last_time.strftime('%Y-%m-%d %H:%M'),
                'exit_price': round(last_close, 5),
                'pnl': round(entry_price - last_close, 5),
                'pnl_pct': round(((entry_price - last_close) / entry_price) * 100, 2),
                'exit_reason': 'End of Data'
            })

    return trades

# =============================================================================
# SUMMARY CALCULATION
# =============================================================================

def calculate_summary(trades: list) -> dict:
    """Calculate summary statistics from trades."""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'net_profit': 0,
            'avg_profit': 0,
            'max_drawdown': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'largest_win': 0,
            'largest_loss': 0
        }

    total = len(trades)
    winning = [t for t in trades if t['pnl'] > 0]
    losing = [t for t in trades if t['pnl'] <= 0]

    net_profit = sum(t['pnl'] for t in trades)

    # Max drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cumulative += t['pnl']
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    largest_win = max([t['pnl'] for t in winning]) if winning else 0
    largest_loss = min([t['pnl'] for t in losing]) if losing else 0

    return {
        'total_trades': total,
        'win_rate': round((len(winning) / total) * 100, 2) if total > 0 else 0,
        'net_profit': round(net_profit, 5),
        'avg_profit': round(net_profit / total, 5) if total > 0 else 0,
        'max_drawdown': round(max_dd, 5),
        'buy_trades': len([t for t in trades if t['direction'] == 'BUY']),
        'sell_trades': len([t for t in trades if t['direction'] == 'SELL']),
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'largest_win': round(largest_win, 5),
        'largest_loss': round(largest_loss, 5)
    }

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.title("📈 Backtesting Tool")
st.caption("EMA Crossover + ATR Trailing Stop")

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("⚙️ Settings")

    # Symbol
    st.subheader("Symbol")
    symbol_map = {
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'BTC/USD': 'BTC-USD',
        'AAPL': 'AAPL'
    }
    symbol_key = st.selectbox(
        "Asset",
        options=list(symbol_map.keys()),
        index=0
    )
    symbol = symbol_map[symbol_key]

    st.divider()

    # Timeframe
    st.subheader("Timeframe")
    timeframe = st.selectbox(
        "Interval",
        options=['15m', '1h', '1d'],
        index=1
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

    # ATR Multiplier
    st.subheader("Risk Management")
    atr_multiplier = st.slider(
        "ATR Multiplier",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1
    )

    st.divider()

    run = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

# =============================================================================
# MAIN CONTENT
# =============================================================================

if run:
    # Fetch data
    with st.spinner(f"Fetching {symbol_key} data..."):
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
            summary = calculate_summary(trades)
        except Exception as e:
            st.error(f"❌ Backtest error: {str(e)}")
            st.stop()

    # Data info
    start_dt = df.index[0].strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'
    end_dt = df.index[-1].strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'
    st.info(f"📊 {len(df)} candles | {start_dt} to {end_dt} | {symbol_key} ({timeframe})")

    st.divider()

    # =============================================================================
    # METRICS
    # =============================================================================

    st.subheader("📊 Performance Summary")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Total Trades", summary['total_trades'])
    m2.metric(
        "Win Rate",
        f"{summary['win_rate']}%",
        delta_color="normal" if summary['win_rate'] >= 50 else "inverse"
    )
    m3.metric(
        "Net Profit",
        f"{summary['net_profit']:.5f}",
        delta_color="normal" if summary['net_profit'] >= 0 else "inverse"
    )
    m4.metric("Max Drawdown", f"{summary['max_drawdown']:.5f}", delta_color="inverse")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Buy Trades", summary['buy_trades'])
    m6.metric("Sell Trades", summary['sell_trades'])
    m7.metric("Winning", summary['winning_trades'], delta_color="normal")
    m8.metric("Losing", summary['losing_trades'], delta_color="inverse")

    st.divider()

    # =============================================================================
    # EQUITY CURVE
    # =============================================================================

    st.subheader("📈 Equity Curve")

    if trades and len(trades) > 0:
        # Build equity data
        equity_data = []
        cumulative = 0
        for trade in trades:
            cumulative += trade['pnl']
            equity_data.append({
                'time': trade['exit_time'],
                'equity': cumulative
            })

        equity_df = pd.DataFrame(equity_data)

        # Plot
        fig = go.Figure()

        # Equity line
        fig.add_trace(go.Scatter(
            x=equity_df['time'],
            y=equity_df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#58a6ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(88, 166, 255, 0.15)'
        ))

        # Zero line
        fig.add_hline(y=0, line_dash="dot", line_color="#8b949e", opacity=0.5)

        fig.update_layout(
            template='plotly_dark',
            hovermode='x unified',
            height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#30363d'),
            yaxis=dict(gridcolor='#30363d')
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
        # Create DataFrame
        trades_df = pd.DataFrame(trades)

        # Rename columns for display
        trades_df_display = trades_df.rename(columns={
            'num': '#',
            'direction': 'Direction',
            'entry_time': 'Entry Time',
            'entry_price': 'Entry Price',
            'exit_time': 'Exit Time',
            'exit_price': 'Exit Price',
            'pnl': 'P&L',
            'pnl_pct': 'P&L %',
            'exit_reason': 'Exit Reason'
        })

        # Display with HTML coloring via markdown
        st.markdown("""
        <style>
            .trade-table { width: 100%; border-collapse: collapse; }
            .trade-table th { background: #161b22; padding: 10px; text-align: left; border-bottom: 1px solid #30363d; color: #8b949e; font-size: 0.85rem; }
            .trade-table td { padding: 10px; border-bottom: 1px solid #30363d; color: #c9d1d9; font-size: 0.9rem; }
            .trade-table tr:hover { background: #161b22; }
            .positive { color: #3fb950; }
            .negative { color: #f85149; }
            .buy-tag { color: #3fb950; font-weight: 600; }
            .sell-tag { color: #f85149; font-weight: 600; }
        </style>
        """, unsafe_allow_html=True)

        # Build HTML table manually (safe, no style.applymap)
        html = '<table class="trade-table">'
        html += '<thead><tr>'
        html += '<th>#</th><th>Direction</th><th>Entry Time</th><th>Entry Price</th>'
        html += '<th>Exit Time</th><th>Exit Price</th><th>P&L</th><th>P&L %</th><th>Exit Reason</th>'
        html += '</tr></thead><tbody>'

        for _, row in trades_df.iterrows():
            pnl_class = 'positive' if row['pnl'] >= 0 else 'negative'
            dir_class = 'buy-tag' if row['direction'] == 'BUY' else 'sell-tag'
            pnl_sign = '+' if row['pnl'] >= 0 else ''

            html += '<tr>'
            html += f"<td>{row['num']}</td>"
            html += f"<td class='{dir_class}'>{row['direction']}</td>"
            html += f"<td>{row['entry_time']}</td>"
            html += f"<td>{row['entry_price']:.5f}</td>"
            html += f"<td>{row['exit_time']}</td>"
            html += f"<td>{row['exit_price']:.5f}</td>"
            html += f"<td class='{pnl_class}'>{pnl_sign}{row['pnl']:.5f}</td>"
            html += f"<td class='{pnl_class}'>{pnl_sign}{row['pnl_pct']:.2f}%</td>"
            html += f"<td>{row['exit_reason']}</td>"
            html += '</tr>'

        html += '</tbody></table>'

        st.markdown(html, unsafe_allow_html=True)

        # CSV Download
        csv_data = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"backtest_{symbol_key.replace('/', '_')}_{timeframe}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ No trades found for this setup. Try a different period or timeframe.")

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
        """)

    with col2:
        st.subheader("📌 Symbols")
        st.markdown("""
        | Asset | Symbol |
        |-------|--------|
        | EUR/USD | EURUSD=X |
        | GBP/USD | GBPUSD=X |
        | BTC/USD | BTC-USD |
        | Apple | AAPL |
        """)

    st.divider()
    st.info("👈 Configure settings in the sidebar and click **Run Backtest**.")
