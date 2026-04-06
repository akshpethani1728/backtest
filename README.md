# Backtesting Tool

A web-based backtesting application for forex/crypto trading strategies using **EMA Crossover with ATR Trailing Stop Loss**.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=Python&logoColor=white)

---

## Features

- **EMA Crossover Strategy**: Uses EMA 20 (fast) and EMA 50 (slow)
- **ATR Trailing Stop**: Dynamic stop loss that trails only in profit direction
- **Pullback Entry**: Enters on confirmation candle after pullback to EMA 20
- **Multiple Assets**: Supports forex, crypto, and stocks via Yahoo Finance
- **Multiple Timeframes**: 15-minute, hourly, and daily data
- **Equity Curve**: Interactive chart of cumulative P&L over time
- **Trade History**: Detailed table of all trades

---

## Live Demo

Deployed on **Streamlit Cloud**: https://your-app.streamlit.app

*(Deploy instructions below)*

---

## Strategy Logic

### Entry Rules

**Buy Signal:**
1. EMA 20 crosses above EMA 50
2. Price closes above both EMAs
3. Wait for pullback to EMA 20
4. Enter on next confirmation candle

**Sell Signal:**
1. EMA 20 crosses below EMA 50
2. Price closes below both EMAs
3. Wait for pullback to EMA 20
4. Enter on next confirmation candle

### Exit Rules

- **Stop Loss**: ATR trailing stop (moves only in profit direction)
- **Exit**: Stop loss hit OR opposite signal appears

---

## Installation (Local)

```bash
# Clone the repo
git clone https://github.com/akshpethani1728/backtest.git
cd backtest

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run streamlit_app.py
```

---

## Deployment to Streamlit Cloud

1. **Fork or push this repo to your GitHub**

2. **Go to [streamlit.io/cloud](https://streamlit.io/cloud)**

3. **Click "New app"**

4. **Select your GitHub repo and branch**

5. **Set:**
   - Main file path: `streamlit_app.py`
   - Python version: `3.11`

6. **Click "Deploy!"**

Your app will be live at `https://[your-username]-backtest.streamlit.app`

---

## Example Symbols

| Asset | Symbol |
|-------|--------|
| EUR/USD | `EURUSD=X` |
| GBP/USD | `GBPUSD=X` |
| USD/JPY | `USDJPY=X` |
| BTC/USD | `BTC-USD` |
| ETH/USD | `ETH-USD` |
| Apple | `AAPL` |
| Tesla | `TSLA` |

---

## Requirements

- Python 3.8+
- streamlit
- pandas
- numpy
- yfinance
- plotly

---

## Disclaimer

This tool is for **educational purposes only**. Past performance does not guarantee future results. Trading involves risk.

---

## License

MIT License
