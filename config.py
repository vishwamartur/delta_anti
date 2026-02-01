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

# Leverage Configuration
DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "200"))  # 200x leverage per trade
AUTO_TOPUP = os.getenv("AUTO_TOPUP", "true").lower() == "true"  # Auto topup to prevent liquidation

# Delta Exchange Fee Configuration (X-Mas Offer Active)
# IMPORTANT: Fees are calculated on NOTIONAL value (spot price × quantity), NOT account balance
FEE_CONFIG = {
    "futures_taker": 0.0005,    # 0.05% of notional (market orders)
    "futures_maker": 0.0002,    # 0.02% of notional (limit orders) <- NOW USING THIS
    "options_rate": 0.0001,     # 0.010% of notional (X-Mas offer)
    "options_max_pct": 0.035,   # Capped at 3.5% of premium
    "order_type": "maker",      # Using LIMIT orders = maker fees (0.02% vs 0.05%)
}

# Trade Frequency Control (to reduce fee accumulation)
MIN_TRADE_INTERVAL_SECONDS = int(os.getenv("MIN_TRADE_INTERVAL", "60"))  # Wait 60s between trades

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
    # Balanced for 200x leverage - gives room for volatility while limiting risk
    # 0.5% move = 100% P/L at 200x, so ~0.8% SL = ~160% risk limit
    "atr_multiplier_tp": 1.2,  # Take profit at 1.2x ATR (~1.5% move)
    "atr_multiplier_sl": 0.8   # Stop loss at 0.8x ATR (~1% move)
}

# Adaptive Trading Configuration (learns from trade history)
ADAPTIVE_TRADING_CONFIG = {
    "enabled": True,                    # Enable/disable adaptive learning
    "min_historical_trades": 5,         # Min trades needed to learn from a combo
    "min_win_rate_threshold": 30,       # Block combos with <30% win rate
    "min_confidence_threshold": 70,     # Raised baseline confidence
    "refresh_interval_minutes": 30,     # How often to re-analyze history
}

# Display Settings
REFRESH_RATE = 1  # Seconds between dashboard updates
CANDLE_HISTORY_SIZE = 100  # Number of candles to keep in memory

# ============ ML Configuration ============
ML_CONFIG = {
    "lstm": {
        "sequence_length": 60,
        "prediction_steps": 5,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2
    },
    "dqn": {
        "state_size": 50,
        "hidden_size": 256,
        "learning_rate": 0.001,
        "gamma": 0.95,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "buffer_size": 10000
    },
    "sentiment": {
        "cache_duration_minutes": 15,
        "news_limit": 50
    }
}

# ============ API Server Configuration ============
API_SERVER_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "debug": os.getenv("API_DEBUG", "false").lower() == "true"
}

# Webhook secrets
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "your-secret-key")

# ============ Trade Manager Configuration ============
INITIAL_ACCOUNT_BALANCE = float(os.getenv("INITIAL_BALANCE", "10000.0"))

# Delta Exchange Fee Structure (Futures)
# Maker: 0.02% | Taker: 0.05%
TRADING_FEES = {
    'maker_fee_pct': 0.02,           # 0.02% maker fee
    'taker_fee_pct': 0.05,           # 0.05% taker fee (market orders)
    'total_round_trip_pct': 0.10,    # Entry (0.05%) + Exit (0.05%) = 0.10%
}

# Minimum profit threshold (must exceed fees to be profitable)
MIN_PROFIT_THRESHOLD_PCT = 0.12  # 0.12% minimum profit after all fees

TRADE_MANAGER_CONFIG = {
    'enable_auto_execution': os.getenv("AUTO_EXECUTION", "false").lower() == "true",
    'enable_trailing_stop': True,
    'max_risk_per_trade': float(os.getenv("MAX_RISK_PER_TRADE", "0.02")),  # 2%
    'max_positions': int(os.getenv("MAX_POSITIONS", "10")),  # Increased to 10
    'max_daily_loss': float(os.getenv("MAX_DAILY_LOSS", "0.50")),  # 50% daily loss limit (raised for testing)
    'max_drawdown': float(os.getenv("MAX_DRAWDOWN", "0.70")),  # 70% max drawdown (from peak)
    'trailing_stop_pct': float(os.getenv("TRAILING_STOP_PCT", "1.5")),  # 1.5%
    'strategy_name': 'delta_anti_v1',
    # Fee structure
    'maker_fee_pct': TRADING_FEES['maker_fee_pct'],
    'taker_fee_pct': TRADING_FEES['taker_fee_pct'],
    'total_fee_pct': TRADING_FEES['total_round_trip_pct'],
    'min_profit_pct': MIN_PROFIT_THRESHOLD_PCT,
    # Sync balance from exchange
    'sync_balance_on_start': True
}

# Create a config object for easy access
class Config:
    """Config wrapper for easy attribute access."""
    API_KEY = API_KEY
    API_SECRET = API_SECRET
    REST_URL = REST_URL
    WS_URL = WS_URL
    TRADING_SYMBOLS = TRADING_SYMBOLS
    DEFAULT_TIMEFRAME = DEFAULT_TIMEFRAME
    RISK_PER_TRADE = RISK_PER_TRADE
    MAX_POSITION_SIZE = MAX_POSITION_SIZE
    MAX_DAILY_TRADES = MAX_DAILY_TRADES
    DEFAULT_LEVERAGE = DEFAULT_LEVERAGE
    AUTO_TOPUP = AUTO_TOPUP
    INDICATOR_CONFIG = INDICATOR_CONFIG
    SIGNAL_CONFIG = SIGNAL_CONFIG
    ML_CONFIG = ML_CONFIG
    API_SERVER_CONFIG = API_SERVER_CONFIG


config = Config()

