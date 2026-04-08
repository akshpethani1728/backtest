//+------------------------------------------------------------------+
//|                                         VantageEMAATR_Strategy.mq5 |
//|                                         Vantage Trading Robot      |
//|                                         EMA Crossover + ATR System |
//+------------------------------------------------------------------+
#property copyright "Vantage Trading System"
#property link      "https://www.vantagefx.com"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| INPUTS - Trade Settings                                           |
//+------------------------------------------------------------------+
input group "=== TRADING ON/OFF ===";
input bool EnableTrading = true;           // Enable/Disable Trading System
input int MagicNumber = 2024;              // Magic Number (ID for this EA)

input group "=== INDICATOR SETTINGS ===";
input int      FastEMA_Period   = 20;      // Fast EMA Period
input int      SlowEMA_Period   = 50;      // Slow EMA Period
input int      ATR_Period       = 14;      // ATR Period
input double   ATR_Multiplier   = 1.5;     // ATR Multiplier (Stop Distance)

input group "=== LOT & RISK ===";
input double   FixedLotSize     = 0.1;      // Fixed Lot Size (0.1 = 10,000 units)
input double   MaxRiskPercent   = 2.0;      // Max Risk Per Trade (% of balance)
input bool     UseFixedLot      = true;      // Use Fixed Lot Size (true) or Risk Percentage (false)

input group "=== ATR REVERSAL SETTINGS ===";
input bool     EnableATRReversal = true;    // Enable ATR Reversal Exit
input double   ATRReversalThreshold = 1.2;   // ATR Expansion Threshold (1.2 = 20%)

input group "=== TRADE MANAGEMENT ===";
input bool     UseTrailingStop   = true;    // Use ATR Trailing Stop
input bool     CloseOnOpposite   = true;    // Close on Opposite Signal
input uint     MaxTradesPerDay   = 5;       // Max Trades Per Day
input int      MaxSlippage       = 3;       // Max Slippage (points)

input group "=== SESSION TIMES (Broker Server Time) ===";
input bool     UseTradingHours   = false;   // Use Trading Hours Filter
input int      TradeStartHour    = 9;       // Start Hour (24h format)
input int      TradeEndHour      = 17;      // End Hour (24h format)

input group "=== ADVANCED ===";
input int      DeviationPoints   = 10;      // Deviation for order execution
input bool     PrintIndicators   = false;   // Print Indicator Values to Log
input bool     VerboseLogging   = true;     // Verbose Logging

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                 |
//+------------------------------------------------------------------+
datetime lastTradeTime = 0;
datetime lastDailyReset = 0;
int tradesToday = 0;
bool eaEnabled = false;

//+------------------------------------------------------------------+
//| ENUMS & STRUCTURES                                               |
//+------------------------------------------------------------------+
enum TradeDirection {
    TRADE_NONE = 0,
    TRADE_BUY  = 1,
    TRADE_SELL = 2
};

struct TradeState {
    TradeDirection direction;
    double        entryPrice;
    double        entryATR;
    double        currentStop;
    datetime      entryTime;
    bool          isActive;
};

//+------------------------------------------------------------------+
//| GLOBAL INSTANCES                                                 |
//+------------------------------------------------------------------+
TradeState currentTrade;
TradeDirection lastDirection = TRADE_NONE;

//+------------------------------------------------------------------+
//| ATR CALCULATION                                                   |
//+------------------------------------------------------------------+
double CalculateATR(int period, int shift = 0) {
    double tr1 = iHigh(_Symbol, PERIOD_CURRENT, shift) - iLow(_Symbol, PERIOD_CURRENT, shift);
    double tr2 = MathAbs(iHigh(_Symbol, PERIOD_CURRENT, shift) - iClose(_Symbol, PERIOD_CURRENT, shift + 1));
    double tr3 = MathAbs(iLow(_Symbol, PERIOD_CURRENT, shift) - iClose(_Symbol, PERIOD_CURRENT, shift + 1));

    double tr = MathMax(tr1, MathMax(tr2, tr3));

    // Use iATR for proper ATR value
    double atr = iATR(_Symbol, PERIOD_CURRENT, period, shift);
    return atr;
}

//+------------------------------------------------------------------+
//| EMA CALCULATION                                                   |
//+------------------------------------------------------------------+
double CalculateEMA(int period, int shift = 0) {
    return iMA(_Symbol, PERIOD_CURRENT, period, 0, MODE_EMA, PRICE_CLOSE, shift);
}

//+------------------------------------------------------------------+
//| CHECK TRADE CONDITIONS                                            |
//+------------------------------------------------------------------+
TradeDirection CheckTradeConditions() {
    double emaFastPrev = CalculateEMA(FastEMA_Period, 2);
    double emaFastCurr = CalculateEMA(FastEMA_Period, 1);
    double emaSlowPrev = CalculateEMA(SlowEMA_Period, 2);
    double emaSlowCurr = CalculateEMA(SlowEMA_Period, 1);

    double currClose = iClose(_Symbol, PERIOD_CURRENT, 1);
    double currATR = CalculateATR(ATR_Period, 1);

    // BUY: Fast EMA crosses above Slow EMA
    if(emaFastPrev <= emaSlowPrev && emaFastCurr > emaSlowCurr) {
        if(currClose > emaFastCurr && currClose > emaSlowCurr) {
            return TRADE_BUY;
        }
    }

    // SELL: Fast EMA crosses below Slow EMA
    if(emaFastPrev >= emaSlowPrev && emaFastCurr < emaSlowCurr) {
        if(currClose < emaFastCurr && currClose < emaSlowCurr) {
            return TRADE_SELL;
        }
    }

    return TRADE_NONE;
}

//+------------------------------------------------------------------+
//| CHECK ATR REVERSAL                                                |
//+------------------------------------------------------------------+
bool CheckATRReversal(TradeDirection tradeDir, double entryATR) {
    if(!EnableATRReversal) return false;

    double currATR = CalculateATR(ATR_Period, 1);
    double currClose = iClose(_Symbol, PERIOD_CURRENT, 1);
    double prevClose = iClose(_Symbol, PERIOD_CURRENT, 2);

    if(currATR > entryATR * ATRReversalThreshold) {
        if(tradeDir == TRADE_BUY && currClose < prevClose) {
            if(VerboseLogging)
                Print("ATR Reversal: BUY - ATR expanded beyond threshold, price moving against");
            return true;
        }
        if(tradeDir == TRADE_SELL && currClose > prevClose) {
            if(VerboseLogging)
                Print("ATR Reversal: SELL - ATR expanded beyond threshold, price moving against");
            return true;
        }
    }

    return false;
}

//+------------------------------------------------------------------+
//| CALCULATE LOT SIZE                                                |
//+------------------------------------------------------------------+
double CalculateLotSize() {
    if(UseFixedLot) {
        return FixedLotSize;
    }

    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * (MaxRiskPercent / 100.0);

    double atr = CalculateATR(ATR_Period, 1);
    double stopDistance = atr * ATR_Multiplier;

    if(stopDistance <= 0) {
        return FixedLotSize;
    }

    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

    if(tickSize == 0) {
        return FixedLotSize;
    }

    double pointValue = tickValue / tickSize;
    double lots = riskAmount / (stopDistance * pointValue);

    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    lots = MathMax(minLot, MathMin(maxLot, MathFloor(lots / stepLot) * stepLot));

    return lots;
}

//+------------------------------------------------------------------+
//| CHECK TRADING HOURS                                               |
//+------------------------------------------------------------------+
bool IsWithinTradingHours() {
    if(!UseTradingHours) return true;

    datetime serverTime = TimeCurrent();
    MqlDateTime dtStruct;
    TimeToStruct(serverTime, dtStruct);

    if(dtStruct.hour >= TradeStartHour && dtStruct.hour < TradeEndHour) {
        return true;
    }

    return false;
}

//+------------------------------------------------------------------+
//| CHECK DAILY TRADE LIMIT                                           |
//+------------------------------------------------------------------+
bool CanOpenTrade() {
    datetime now = TimeCurrent();
    MqlDateTime today;
    TimeToStruct(now, today);

    string dateKey = StringFormat("%04d%02d%02d", today.year, today.mon, today.day);

    if(lastDailyReset != StringToTime(dateKey)) {
        tradesToday = 0;
        lastDailyReset = StringToTime(dateKey);
    }

    if(tradesToday >= MaxTradesPerDay) {
        if(VerboseLogging)
            Print("Daily trade limit reached: ", MaxTradesPerDay);
        return false;
    }

    return true;
}

//+------------------------------------------------------------------+
//| OPEN TRADE                                                        |
//+------------------------------------------------------------------+
bool OpenTrade(TradeDirection tradeDir) {
    if(tradeDir == TRADE_NONE) return false;
    if(!CanOpenTrade()) return false;
    if(!IsWithinTradingHours()) return false;
    if(!EnableTrading) return false;

    double lotSize = CalculateLotSize();
    double atr = CalculateATR(ATR_Period, 1);
    double stopDistance = atr * ATR_Multiplier;

    double askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bidPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);

    double entryPrice = 0;
    double stopLoss = 0;

    ENUM_ORDER_TYPE orderType;
    ENUM_ORDER_TYPE_FILLING fillingType = ORDER_FILLING_FOK;

    if(tradeDir == TRADE_BUY) {
        entryPrice = askPrice;
        orderType = ORDER_TYPE_BUY;
        stopLoss = entryPrice - stopDistance;

        double minStop = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
        if(stopLoss < bidPrice - minStop) {
            stopLoss = bidPrice - minStop;
        }
    } else if(tradeDir == TRADE_SELL) {
        entryPrice = bidPrice;
        orderType = ORDER_TYPE_SELL;
        stopLoss = entryPrice + stopDistance;

        double minStop = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
        if(stopLoss > askPrice + minStop) {
            stopLoss = askPrice + minStop;
        }
    }

    MqlTradeRequest request = {};
    MqlTradeResult result = {};

    request.action = TRADE_ACTION_DEAL;
    request.magic = MagicNumber;
    request.symbol = _Symbol;
    request.volume = lotSize;
    request.type = orderType;
    request.price = entryPrice;
    request.sl = stopLoss;
    request.tp = 0;
    request.deviation = DeviationPoints;
    request.type_filling = fillingType;
    request.comment = "VantageEMAATR";

    bool success = OrderSend(request, result);

    if(success && result.retcode == TRADE_RETCODE_DONE) {
        currentTrade.direction = tradeDir;
        currentTrade.entryPrice = entryPrice;
        currentTrade.entryATR = atr;
        currentTrade.currentStop = stopLoss;
        currentTrade.entryTime = TimeCurrent();
        currentTrade.isActive = true;

        tradesToday++;
        lastTradeTime = TimeCurrent();
        lastDirection = tradeDir;

        if(VerboseLogging) {
            Print("Trade OPENED: ", tradeDir == TRADE_BUY ? "BUY" : "SELL",
                  " at ", entryPrice,
                  " Lots: ", lotSize,
                  " Stop: ", stopLoss,
                  " ATR: ", atr);
        }

        return true;
    } else {
        if(VerboseLogging) {
            Print("Trade FAILED: ", result.comment,
                  " Retcode: ", result.retcode);
        }
        return false;
    }
}

//+------------------------------------------------------------------+
//| CLOSE TRADE                                                      |
//+------------------------------------------------------------------+
bool CloseTrade(string reason) {
    if(!currentTrade.isActive) return false;

    if(!PositionSelect(_Symbol)) {
        currentTrade.isActive = false;
        currentTrade.direction = TRADE_NONE;
        return false;
    }

    double lotSize = PositionGetDouble(POSITION_VOLUME);
    ENUM_ORDER_TYPE orderType = (currentTrade.direction == TRADE_BUY) ?
                               ORDER_TYPE_SELL : ORDER_TYPE_BUY;

    double closePrice = (currentTrade.direction == TRADE_BUY) ?
                       SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                       SymbolInfoDouble(_Symbol, SYMBOL_ASK);

    MqlTradeRequest request = {};
    MqlTradeResult result = {};

    request.action = TRADE_ACTION_DEAL;
    request.magic = MagicNumber;
    request.symbol = _Symbol;
    request.volume = lotSize;
    request.type = orderType;
    request.price = closePrice;
    request.deviation = DeviationPoints;
    request.comment = reason;

    bool success = OrderSend(request, result);

    if(success && result.retcode == TRADE_RETCODE_DONE) {
        if(VerboseLogging) {
            double pnl = result.profit;
            Print("Trade CLOSED: ", reason,
                  " P&L: ", pnl);
        }

        currentTrade.isActive = false;
        currentTrade.direction = TRADE_NONE;
        return true;
    } else {
        if(VerboseLogging) {
            Print("Close FAILED: ", result.comment);
        }
        return false;
    }
}

//+------------------------------------------------------------------+
//| UPDATE TRAILING STOP                                              |
//+------------------------------------------------------------------+
void UpdateTrailingStop() {
    if(!currentTrade.isActive) return;
    if(!UseTrailingStop) return;

    double currATR = CalculateATR(ATR_Period, 1);
    double currClose = iClose(_Symbol, PERIOD_CURRENT, 1);

    if(currentTrade.direction == TRADE_BUY) {
        double newStop = currClose - (currATR * ATR_Multiplier);

        if(newStop > currentTrade.currentStop) {
            if(!PositionSelect(_Symbol)) return;

            MqlTradeRequest request = {};
            MqlTradeResult result = {};

            request.action = TRADE_ACTION_SLLOW;
            request.magic = MagicNumber;
            request.symbol = _Symbol;
            request.sl = newStop;
            request.position = PositionGetInteger(POSITION_TICKET);

            if(OrderSend(request, result) && result.retcode == TRADE_RETCODE_DONE) {
                currentTrade.currentStop = newStop;

                if(VerboseLogging) {
                    Print("Trailing Stop UPDATED: BUY - New Stop: ", newStop);
                }
            }
        }
    } else if(currentTrade.direction == TRADE_SELL) {
        double newStop = currClose + (currATR * ATR_Multiplier);

        if(newStop < currentTrade.currentStop) {
            if(!PositionSelect(_Symbol)) return;

            MqlTradeRequest request = {};
            MqlTradeResult result = {};

            request.action = TRADE_ACTION_SLLOW;
            request.magic = MagicNumber;
            request.symbol = _Symbol;
            request.sl = newStop;
            request.position = PositionGetInteger(POSITION_TICKET);

            if(OrderSend(request, result) && result.retcode == TRADE_RETCODE_DONE) {
                currentTrade.currentStop = newStop;

                if(VerboseLogging) {
                    Print("Trailing Stop UPDATED: SELL - New Stop: ", newStop);
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| CHECK EXISTING POSITIONS                                          |
//+------------------------------------------------------------------+
void CheckExistingPositions() {
    if(PositionSelect(_Symbol)) {
        currentTrade.direction = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ?
                                  TRADE_BUY : TRADE_SELL;
        currentTrade.entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
        currentTrade.entryTime = (datetime)PositionGetInteger(POSITION_TIME);
        currentTrade.currentStop = PositionGetDouble(POSITION_SL);
        currentTrade.isActive = true;

        double atr = CalculateATR(ATR_Period, 1);
        currentTrade.entryATR = atr;
    } else {
        currentTrade.isActive = false;
        currentTrade.direction = TRADE_NONE;
    }
}

//+------------------------------------------------------------------+
//| EXPERT ON TICKET                                                  |
//+------------------------------------------------------------------+
void OnTick() {
    if(!EnableTrading) {
        if(currentTrade.isActive) {
            CloseTrade("EA Disabled");
        }
        return;
    }

    CheckExistingPositions();

    if(currentTrade.isActive) {
        UpdateTrailingStop();

        double currClose = iClose(_Symbol, PERIOD_CURRENT, 1);
        double currLow = iLow(_Symbol, PERIOD_CURRENT, 1);
        double currHigh = iHigh(_Symbol, PERIOD_CURRENT, 1);

        if(currentTrade.direction == TRADE_BUY) {
            double stopLevel = currentTrade.currentStop;

            if(currLow <= stopLevel) {
                CloseTrade("ATR Stop Hit");
                return;
            }

            if(CheckATRReversal(TRADE_BUY, currentTrade.entryATR)) {
                CloseTrade("ATR Reversal");
                return;
            }

            if(CloseOnOpposite) {
                TradeDirection signal = CheckTradeConditions();
                if(signal == TRADE_SELL) {
                    CloseTrade("Opposite Signal");
                    return;
                }
            }
        }
        else if(currentTrade.direction == TRADE_SELL) {
            double stopLevel = currentTrade.currentStop;

            if(currHigh >= stopLevel) {
                CloseTrade("ATR Stop Hit");
                return;
            }

            if(CheckATRReversal(TRADE_SELL, currentTrade.entryATR)) {
                CloseTrade("ATR Reversal");
                return;
            }

            if(CloseOnOpposite) {
                TradeDirection signal = CheckTradeConditions();
                if(signal == TRADE_BUY) {
                    CloseTrade("Opposite Signal");
                    return;
                }
            }
        }
    }
    else {
        TradeDirection signal = CheckTradeConditions();

        if(signal != TRADE_NONE) {
            OpenTrade(signal);
        }
    }
}

//+------------------------------------------------------------------+
//| EXPERT ON INIT                                                    |
//+------------------------------------------------------------------+
int OnInit() {
    eaEnabled = EnableTrading;

    if(VerboseLogging) {
        Print("========================================");
        Print("Vantage EMA ATR Trading System");
        Print("EA Initialized on: ", _Symbol);
        Print("Fast EMA: ", FastEMA_Period);
        Print("Slow EMA: ", SlowEMA_Period);
        Print("ATR Period: ", ATR_Period);
        Print("ATR Multiplier: ", ATR_Multiplier);
        Print("ATR Reversal: ", EnableATRReversal);
        Print("Lot Size: ", FixedLotSize);
        Print("Trading Enabled: ", EnableTrading);
        Print("========================================");
    }

    CheckExistingPositions();

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| EXPERT ON DEINIT                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    if(VerboseLogging) {
        Print("EA Deinitialized. Reason: ", reason);
    }
}

//+------------------------------------------------------------------+
//| EXPERT ON CALCULATE                                                |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
               const int prev_calculated,
               const datetime &time[],
               const double &open[],
               const double &high[],
               const double &low[],
               const double &close[],
               const long &tick_volume[],
               const long &volume[],
               const double &spread[]) {
    return rates_total;
}
//+------------------------------------------------------------------+
