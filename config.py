"""
Delta Exchange Trading System Configuration
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv("DELTA_API_KEY", "")
API_SECRET = os.getenv("DELTA_API_SECRET", "")
REST_URL = os.getenv("DELTA_REST_URL", "https://api.india.delta.exchange")
WS_URL = os.getenv("DELTA_WS_URL", "wss://socket.india.delta.exchange")

# Trading Symbols
TRADING_SYMBOLS = os.getenv("TRADING_SYMBOLS", "BTCUSD,ETHUSD").split(",")
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "5m")

# Risk Management
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))
MAX_POSITION_SIZE = int(os.getenv("MAX_POSITION_SIZE", "100"))
MAX_DAILY_TRADES = 10
MAX_DRAWDOWN_PERCENT = 0.05

# Technical Indicator Settings
INDICATOR_CONFIG = {
    "rsi": {
        "period": int(os.getenv("RSI_PERIOD", "14")),
        "overbought": 70,
        "oversold": 30
    },
    "macd": {
        "fast": int(os.getenv("MACD_FAST", "12")),
        "slow": int(os.getenv("MACD_SLOW", "26")),
        "signal": int(os.getenv("MACD_SIGNAL", "9"))
    },
    "bollinger": {
        "period": int(os.getenv("BB_PERIOD", "20")),
        "std_dev": float(os.getenv("BB_STD", "2"))
    },
    "ema": {
        "short": 9,
        "medium": 21,
        "long": 50
    },
    "atr": {
        "period": 14
    }
}

# Candle Resolutions
VALID_RESOLUTIONS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]

# WebSocket Channels
WS_CHANNELS = {
    "ticker": "v2/ticker",
    "candlestick": "candlestick_",  # Append resolution e.g., candlestick_5m
    "orderbook": "l2_orderbook",
    "trades": "all_trades",
    "mark_price": "mark_price"
}

# Signal Configuration
SIGNAL_CONFIG = {
    "min_confidence": 60,  # Minimum confidence score to suggest trade (0-100)
    "confirm_candles": 2,   # Number of candles to confirm signal
    "atr_multiplier_tp": 2.0,  # Take profit at 2x ATR
    "atr_multiplier_sl": 1.5   # Stop loss at 1.5x ATR
}

# Display Settings
REFRESH_RATE = 1  # Seconds between dashboard updates
CANDLE_HISTORY_SIZE = 100  # Number of candles to keep in memory
