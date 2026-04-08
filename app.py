"""
Forex/Crypto Backtesting Application
EMA Crossover with ATR Trailing Stop Loss Strategy
"""

import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import pandas as pd
import yfinance as yf
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_data(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance.
    Returns DataFrame with OHLCV data or raises ValueError on failure.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for symbol '{symbol}'. Please check the symbol and date range.")

        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Drop rows with NaN in essential columns
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        df = df[df['Close'] > 0]

        return df

    except Exception as e:
        if "No data" in str(e) or "404" in str(e):
            raise ValueError(f"Invalid symbol '{symbol}'. Please use a valid Yahoo Finance symbol (e.g., EURUSD=X, BTC-USD).")
        elif "network" in str(e).lower() or "connection" in str(e).lower():
            raise ValueError("Network error. Please check your internet connection and try again.")
        else:
            raise ValueError(f"Failed to fetch data: {str(e)}")


# =============================================================================
# INDICATORS
# =============================================================================

def calculate_indicators(df: pd.DataFrame, ema_fast: int = 20, ema_slow: int = 50, atr_period: int = 14) -> pd.DataFrame:
    """
    Calculate EMA 20, EMA 50, and ATR indicators.
    """
    df = df.copy()

    # Calculate EMAs
    df['EMA_Fast'] = df['Close'].ewm(span=ema_fast, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=ema_slow, adjust=False).mean()

    # Calculate ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=atr_period).mean()

    # Fill NaN values at the beginning
    df = df.bfill().ffill()

    return df


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def run_backtest(df: pd.DataFrame, atr_multiplier: float = 1.5) -> dict:
    """
    Run the backtesting engine with EMA crossover + ATR trailing stop strategy.

    Buy Conditions:
    - EMA 20 crosses above EMA 50
    - Price closes above both EMAs
    - After crossover, wait for pullback to EMA 20
    - Enter buy on confirmation candle after pullback

    Sell Conditions:
    - EMA 20 crosses below EMA 50
    - Price closes below both EMAs
    - After crossover, wait for pullback to EMA 20
    - Enter sell on confirmation candle after pullback

    Stop Loss: ATR trailing stop (never moves backward in profit)
    Exit: Stop loss hit OR opposite signal
    """
    df = df.copy()
    df = calculate_indicators(df)

    trades = []
    position = None  # None, 'buy', or 'sell'
    entry_price = 0
    entry_time = None
    atr_trailing_stop = 0
    in_pullback = False
    pullback_crossover_idx = None
    pullback_price = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        prev_ema_fast = df.iloc[i - 1]['EMA_Fast'] if i > 0 else row['EMA_Fast']
        prev_ema_slow = df.iloc[i - 1]['EMA_Slow'] if i > 0 else row['EMA_Slow']
        prev_atr = df.iloc[i - 1]['ATR'] if i > 0 else row['ATR']

        curr_ema_fast = row['EMA_Fast']
        curr_ema_slow = row['EMA_Slow']
        curr_close = row['Close']
        curr_high = row['High']
        curr_low = row['Low']
        curr_atr = row['ATR']

        if pd.isna(curr_ema_fast) or pd.isna(curr_ema_slow) or pd.isna(curr_atr):
            continue

        # Detect crossover
        curr_crossover_up = (prev_ema_fast <= prev_ema_slow and curr_ema_fast > curr_ema_slow)
        curr_crossover_down = (prev_ema_fast >= prev_ema_slow and curr_ema_fast < curr_ema_slow)

        # === CLOSE EXISTING POSITION ===
        if position == 'buy':
            # Update ATR trailing stop (only moves up, never down)
            new_stop = curr_close - (atr_multiplier * curr_atr)
            if new_stop > atr_trailing_stop:
                atr_trailing_stop = new_stop

            # Check stop loss hit
            if curr_low <= atr_trailing_stop:
                exit_price = atr_trailing_stop
                trades.append({
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'entry_price': round(entry_price, 5),
                    'exit_time': idx.strftime('%Y-%m-%d %H:%M'),
                    'exit_price': round(exit_price, 5),
                    'direction': 'BUY',
                    'pnl': round(exit_price - entry_price, 5),
                    'pnl_pct': round((exit_price / entry_price - 1) * 100, 2),
                    'exit_reason': 'Stop Loss'
                })
                position = None
                atr_trailing_stop = 0
                in_pullback = False

            # Check opposite signal (sell)
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
            # Update ATR trailing stop (only moves down, never up)
            new_stop = curr_close + (atr_multiplier * curr_atr)
            if new_stop < atr_trailing_stop:
                atr_trailing_stop = new_stop

            # Check stop loss hit
            if curr_high >= atr_trailing_stop:
                exit_price = atr_trailing_stop
                trades.append({
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M'),
                    'entry_price': round(entry_price, 5),
                    'exit_time': idx.strftime('%Y-%m-%d %H:%M'),
                    'exit_price': round(exit_price, 5),
                    'direction': 'SELL',
                    'pnl': round(entry_price - exit_price, 5),
                    'pnl_pct': round((1 - exit_price / entry_price) * 100, 2),
                    'exit_reason': 'Stop Loss'
                })
                position = None
                atr_trailing_stop = 0
                in_pullback = False

            # Check opposite signal (buy)
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

        # === LOOK FOR NEW ENTRY (only if no position) ===
        if position is None:
            # BUY SIGNAL
            if curr_crossover_up and curr_close > curr_ema_fast and curr_close > curr_ema_slow:
                # Mark crossover, wait for pullback
                in_pullback = True
                pullback_crossover_idx = i
                pullback_price = curr_close

            # Check for pullback entry on BUY
            elif in_pullback:
                # Price pulled back to EMA 20 (touched or came close)
                pullback_tolerance = curr_atr * 0.5
                near_ema = curr_low <= (curr_ema_fast + pullback_tolerance) and curr_low >= (curr_ema_fast - pullback_tolerance * 3)

                if near_ema and curr_close > curr_ema_fast and curr_close > curr_ema_slow:
                    # Confirmation candle - price moved back above EMA 20
                    position = 'buy'
                    entry_price = curr_close
                    entry_time = idx
                    atr_trailing_stop = curr_close - (atr_multiplier * curr_atr)
                    in_pullback = False
                elif curr_close < curr_ema_slow:
                    # Price dropped below slow EMA, cancel the buy setup
                    in_pullback = False

            # SELL SIGNAL
            elif curr_crossover_down and curr_close < curr_ema_fast and curr_close < curr_ema_slow:
                # Mark crossover, wait for pullback
                in_pullback = True
                pullback_crossover_idx = i
                pullback_price = curr_close

            # Check for pullback entry on SELL
            elif in_pullback:
                pullback_tolerance = curr_atr * 0.5
                near_ema = curr_high >= (curr_ema_fast - pullback_tolerance) and curr_high <= (curr_ema_fast + pullback_tolerance * 3)

                if near_ema and curr_close < curr_ema_fast and curr_close < curr_ema_slow:
                    # Confirmation candle - price moved back below EMA 20
                    position = 'sell'
                    entry_price = curr_close
                    entry_time = idx
                    atr_trailing_stop = curr_close + (atr_multiplier * curr_atr)
                    in_pullback = False
                elif curr_close > curr_ema_slow:
                    # Price rose above slow EMA, cancel the sell setup
                    in_pullback = False

    # Close any open position at the end
    if position is not None and len(df) > 0:
        last_row = df.iloc[-1]
        last_close = last_row['Close']
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

    return {'trades': trades}


def calculate_summary(trades: list) -> dict:
    """
    Calculate summary statistics from trades.
    """
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
            'losing_trades': 0
        }

    total = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    win_rate = (len(wins) / total) * 100 if total > 0 else 0

    net_profit = sum(t['pnl'] for t in trades)
    avg_profit = net_profit / total if total > 0 else 0

    # Calculate max drawdown
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
        'win_rate': round(win_rate, 2),
        'net_profit': round(net_profit, 5),
        'avg_profit': round(avg_profit, 5),
        'max_drawdown': round(max_dd, 5),
        'buy_trades': len([t for t in trades if t['direction'] == 'BUY']),
        'sell_trades': len([t for t in trades if t['direction'] == 'SELL']),
        'winning_trades': len(wins),
        'losing_trades': len(losses)
    }


# =============================================================================
# VALIDATION
# =============================================================================

def validate_inputs(symbol: str, start_date: str, end_date: str, timeframe: str, atr_multiplier: str) -> tuple:
    """
    Validate all user inputs.
    Returns (is_valid, error_message).
    """
    # Symbol validation
    if not symbol or not symbol.strip():
        return False, "Please enter a symbol."

    symbol = symbol.strip()

    # Timeframe validation
    valid_timeframes = ['15m', '1h', '1d']
    if timeframe not in valid_timeframes:
        return False, f"Invalid timeframe. Choose from: {', '.join(valid_timeframes)}"

    # Date validation
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        if start >= end:
            return False, "End date must be after start date."

        # Check date range is reasonable
        delta = (end - start).days
        if timeframe == '15m' and delta > 60:
            return False, "For 15-minute data, date range should not exceed 60 days."
        elif timeframe == '1h' and delta > 730:
            return False, "For hourly data, date range should not exceed 2 years."
        elif timeframe == '1d' and delta > 3650:
            return False, "For daily data, date range should not exceed 10 years."

    except ValueError:
        return False, "Invalid date format. Use YYYY-MM-DD."

    # ATR multiplier validation
    try:
        atr_mult = float(atr_multiplier)
        if atr_mult <= 0:
            return False, "ATR multiplier must be greater than 0."
        if atr_mult > 10:
            return False, "ATR multiplier seems too high. Use a value between 0.5 and 10."
    except (ValueError, TypeError):
        return False, "ATR multiplier must be a number."

    return True, None


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/backtest', methods=['POST'])
def backtest():
    """
    Run backtest endpoint.
    Expects JSON: {symbol, start_date, end_date, timeframe, atr_multiplier}
    """
    try:
        data = request.get_json()

        symbol = data.get('symbol', '')
        start_date = data.get('start_date', '')
        end_date = data.get('end_date', '')
        timeframe = data.get('timeframe', '1d')
        atr_multiplier = data.get('atr_multiplier', '1.5')

        # Validate inputs
        is_valid, error = validate_inputs(symbol, start_date, end_date, timeframe, atr_multiplier)
        if not is_valid:
            return jsonify({'success': False, 'error': error}), 400

        # Fetch data
        df = fetch_data(symbol, start_date, end_date, timeframe)

        if len(df) < 100:
            return jsonify({
                'success': False,
                'error': f"Not enough data. Got {len(df)} candles. Need at least 100."
            }), 400

        # Run backtest
        result = run_backtest(df, float(atr_multiplier))
        summary = calculate_summary(result['trades'])

        # Calculate equity curve data
        equity_curve = []
        cumulative = 0
        for trade in result['trades']:
            cumulative += trade['pnl']
            equity_curve.append({
                'time': trade['exit_time'],
                'equity': round(cumulative, 5)
            })

        return jsonify({
            'success': True,
            'summary': summary,
            'trades': result['trades'],
            'equity_curve': equity_curve,
            'data_points': len(df)
        })

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  Backtesting Application")
    print("  EMA Crossover + ATR Trailing Stop Strategy")
    print("=" * 60)
    print("\n  Starting server at http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop\n")
    app.run(debug=True, host='127.0.0.1', port=5000)
