//+------------------------------------------------------------------+
//|                              VantageEMAATR_Strategy.mq5           |
//|                                         Vantage Trading Robot     |
//+------------------------------------------------------------------+
#property copyright "Vantage Trading System"
#property link      "https://www.vantagefx.com"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input bool   EnableTrading      = true;      // Trading System ON/OFF
input int     MagicNumber        = 2024;      // Magic Number
input int     FastEMAPeriod      = 20;        // Fast EMA Period
input int     SlowEMAPeriod      = 50;        // Slow EMA Period
input int     ATRPeriod          = 14;        // ATR Period
input double  ATRMultiplier     = 1.5;       // ATR Multiplier for Stop
input double  FixedLotSize      = 0.1;       // Fixed Lot Size
input bool    UseTrailingStop   = true;      // Use ATR Trailing Stop
input bool    CloseOnOpposite   = true;      // Close on Opposite Signal
input bool    EnableATRReversal = true;      // Enable ATR Reversal Exit
input double  ATRReversalThreshold = 1.2;     // ATR Reversal Threshold
input uint    MaxTradesPerDay   = 5;         // Max Trades Per Day

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+
datetime   g_lastTradeTime      = 0;
datetime   g_lastDailyReset     = 0;
int        g_tradesToday        = 0;
bool       g_tradeActive        = false;
double     g_entryPrice         = 0;
double     g_entryATR           = 0;
double     g_currentStop         = 0;
datetime   g_entryTime          = 0;
bool       g_isBuyTrade         = false;

//+------------------------------------------------------------------+
//| Get ATR Value                                                     |
//+------------------------------------------------------------------+
double GetATR(int period, int shift) {
    double atrArr[];
    ArraySetAsSeries(atrArr, true);
    int handle = iATR(_Symbol, PERIOD_CURRENT, period);
    if(handle == INVALID_HANDLE) return 0;
    CopyBuffer(handle, 0, shift, 1, atrArr);
    IndicatorRelease(handle);
    return atrArr[0];
}
//+------------------------------------------------------------------+
//| Get EMA Value                                                     |
//+------------------------------------------------------------------+
double GetEMA(int period, int shift) {
    double emaArr[];
    ArraySetAsSeries(emaArr, true);
    int handle = iMA(_Symbol, PERIOD_CURRENT, period, 0, MODE_EMA, PRICE_CLOSE);
    if(handle == INVALID_HANDLE) return 0;
    CopyBuffer(handle, 0, shift, 1, emaArr);
    IndicatorRelease(handle);
    return emaArr[0];
}

//+------------------------------------------------------------------+
//| Check if EMA Crossover Signal                                     |
//+------------------------------------------------------------------+
int CheckSignal() {
    double emaFast1 = GetEMA(FastEMAPeriod, 1);
    double emaFast2 = GetEMA(FastEMAPeriod, 2);
    double emaSlow1 = GetEMA(SlowEMAPeriod, 1);
    double emaSlow2 = GetEMA(SlowEMAPeriod, 2);
    double closeArr[];
    ArraySetAsSeries(closeArr, true);
    CopyClose(_Symbol, PERIOD_CURRENT, 1, 1, closeArr);
    double close = closeArr[0];

    // BUY: EMA Fast crosses above EMA Slow
    if(emaFast2 <= emaSlow2 && emaFast1 > emaSlow1) {
        if(close > emaFast1 && close > emaSlow1) {
            return 1; // BUY
        }
    }

    // SELL: EMA Fast crosses below EMA Slow
    if(emaFast2 >= emaSlow2 && emaFast1 < emaSlow1) {
        if(close < emaFast1 && close < emaSlow1) {
            return -1; // SELL
        }
    }

    return 0; // NO SIGNAL
}

//+------------------------------------------------------------------+
//| Check ATR Reversal                                                |
//+------------------------------------------------------------------+
bool CheckATRReversal() {
    if(!EnableATRReversal) return false;

    double currATR = GetATR(ATRPeriod, 1);
    double prevATR = GetATR(ATRPeriod, 2);
    double closeArr1[];
    double closeArr2[];
    ArraySetAsSeries(closeArr1, true);
    ArraySetAsSeries(closeArr2, true);
    CopyClose(_Symbol, PERIOD_CURRENT, 1, 1, closeArr1);
    CopyClose(_Symbol, PERIOD_CURRENT, 2, 1, closeArr2);
    double currClose = closeArr1[0];
    double prevClose = closeArr2[0];

    // ATR expanding and price moving against us
    if(currATR > g_entryATR * ATRReversalThreshold) {
        if(g_isBuyTrade && currClose < prevClose) return true;
        if(!g_isBuyTrade && currClose > prevClose) return true;
    }

    return false;
}

//+------------------------------------------------------------------+
//| Can Open Trade                                                    |
//+------------------------------------------------------------------+
bool CanOpenTrade() {
    datetime now = TimeCurrent();
    MqlDateTime dt;
    TimeToStruct(now, dt);

    string dateStr = StringFormat("%04d%02d%02d", dt.year, dt.mon, dt.day);

    if(g_lastDailyReset != StringToTime(dateStr)) {
        g_tradesToday = 0;
        g_lastDailyReset = StringToTime(dateStr);
    }

    if(g_tradesToday >= MaxTradesPerDay) return false;

    return true;
}

//+------------------------------------------------------------------+
//| Open Trade                                                        |
//+------------------------------------------------------------------+
bool OpenTrade(bool isBuy) {
    if(!EnableTrading) return false;
    if(!CanOpenTrade()) return false;

    double lotSize = FixedLotSize;
    double atr = GetATR(ATRPeriod, 1);
    double stopDist = atr * ATRMultiplier;

    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double price = isBuy ? ask : bid;
    double stopLoss = isBuy ? price - stopDist : price + stopDist;

    MqlTradeRequest request = {};
    MqlTradeResult result = {};

    request.action = TRADE_ACTION_DEAL;
    request.magic = MagicNumber;
    request.symbol = _Symbol;
    request.volume = lotSize;
    request.type = isBuy ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
    request.price = price;
    request.sl = stopLoss;
    request.deviation = 10;
    request.type_filling = ORDER_FILLING_FOK;
    request.comment = "VantageEMA";

    bool success = OrderSend(request, result);

    if(success && result.retcode == TRADE_RETCODE_DONE) {
        g_tradeActive = true;
        g_isBuyTrade = isBuy;
        g_entryPrice = price;
        g_entryATR = atr;
        g_currentStop = stopLoss;
        g_entryTime = TimeCurrent();
        g_tradesToday++;
        Print("Trade Opened: ", isBuy ? "BUY" : "SELL", " at ", price, " SL: ", stopLoss);
        return true;
    } else {
        Print("Order Failed: ", result.comment);
        return false;
    }
}

//+------------------------------------------------------------------+
//| Close Trade                                                       |
//+------------------------------------------------------------------+
bool CloseTrade(string reason) {
    if(!g_tradeActive) return false;

    if(!PositionSelect(_Symbol)) {
        g_tradeActive = false;
        return true;
    }

    double volume = PositionGetDouble(POSITION_VOLUME);
    ENUM_ORDER_TYPE closeType = g_isBuyTrade ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
    double closePrice = g_isBuyTrade ? SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                                       SymbolInfoDouble(_Symbol, SYMBOL_ASK);

    MqlTradeRequest request = {};
    MqlTradeResult result = {};

    request.action = TRADE_ACTION_DEAL;
    request.magic = MagicNumber;
    request.symbol = _Symbol;
    request.volume = volume;
    request.type = closeType;
    request.price = closePrice;
    request.deviation = 10;
    request.comment = reason;

    bool success = OrderSend(request, result);

    if(success && result.retcode == TRADE_RETCODE_DONE) {
        Print("Trade Closed: ", reason, " Profit: ", result.profit);
        g_tradeActive = false;
        return true;
    }

    return false;
}

//+------------------------------------------------------------------+
//| Update Trailing Stop                                              |
//+------------------------------------------------------------------+
void UpdateTrailingStop() {
    if(!g_tradeActive) return;
    if(!UseTrailingStop) return;

    double atr = GetATR(ATRPeriod, 1);
    double closeArr[];
    ArraySetAsSeries(closeArr, true);
    CopyClose(_Symbol, PERIOD_CURRENT, 1, 1, closeArr);
    double close = closeArr[0];

    if(g_isBuyTrade) {
        double newStop = close - (atr * ATRMultiplier);
        if(newStop > g_currentStop) {
            if(PositionSelect(_Symbol)) {
                MqlTradeRequest request = {};
                MqlTradeResult result = {};
                request.action = TRADE_ACTION_SLLOW;
                request.magic = MagicNumber;
                request.symbol = _Symbol;
                request.sl = newStop;
                request.position = PositionGetInteger(POSITION_TICKET);
                OrderSend(request, result);
                g_currentStop = newStop;
            }
        }
    } else {
        double newStop = close + (atr * ATRMultiplier);
        if(newStop < g_currentStop) {
            if(PositionSelect(_Symbol)) {
                MqlTradeRequest request = {};
                MqlTradeResult result = {};
                request.action = TRADE_ACTION_SLLOW;
                request.magic = MagicNumber;
                request.symbol = _Symbol;
                request.sl = newStop;
                request.position = PositionGetInteger(POSITION_TICKET);
                OrderSend(request, result);
                g_currentStop = newStop;
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Check Existing Position                                           |
//+------------------------------------------------------------------+
void CheckExistingPosition() {
    if(PositionSelect(_Symbol)) {
        g_tradeActive = true;
        g_isBuyTrade = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
        g_entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
        g_entryTime = (datetime)PositionGetInteger(POSITION_TIME);
        g_currentStop = PositionGetDouble(POSITION_SL);
        g_entryATR = GetATR(ATRPeriod, 1);
    } else {
        g_tradeActive = false;
    }
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick() {
    // Check if trading is disabled
    if(!EnableTrading) {
        if(g_tradeActive) {
            CloseTrade("EA Disabled");
        }
        return;
    }

    // Check existing position
    CheckExistingPosition();

    if(g_tradeActive) {
        // Update trailing stop
        UpdateTrailingStop();

        double lowArr[];
        double highArr[];
        ArraySetAsSeries(lowArr, true);
        ArraySetAsSeries(highArr, true);
        CopyLow(_Symbol, PERIOD_CURRENT, 1, 1, lowArr);
        CopyHigh(_Symbol, PERIOD_CURRENT, 1, 1, highArr);
        double low = lowArr[0];
        double high = highArr[0];

        // Check stop loss
        if(g_isBuyTrade) {
            if(low <= g_currentStop) {
                CloseTrade("Stop Loss");
                return;
            }
        } else {
            if(high >= g_currentStop) {
                CloseTrade("Stop Loss");
                return;
            }
        }

        // Check ATR reversal
        if(CheckATRReversal()) {
            CloseTrade("ATR Reversal");
            return;
        }

        // Check opposite signal
        if(CloseOnOpposite) {
            int signal = CheckSignal();
            if(g_isBuyTrade && signal == -1) {
                CloseTrade("Opposite Signal");
                return;
            }
            if(!g_isBuyTrade && signal == 1) {
                CloseTrade("Opposite Signal");
                return;
            }
        }

    } else {
        // Check for new signal
        int signal = CheckSignal();
        if(signal == 1) {
            OpenTrade(true); // BUY
        } else if(signal == -1) {
            OpenTrade(false); // SELL
        }
    }
}

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit() {
    Print("========================================");
    Print("Vantage EMA ATR Trading System");
    Print("Symbol: ", _Symbol);
    Print("Enable Trading: ", EnableTrading);
    Print("Fast EMA: ", FastEMAPeriod);
    Print("Slow EMA: ", SlowEMAPeriod);
    Print("ATR Period: ", ATRPeriod);
    Print("ATR Multiplier: ", ATRMultiplier);
    Print("========================================");

    CheckExistingPosition();
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    Print("EA Removed. Reason: ", reason);
}