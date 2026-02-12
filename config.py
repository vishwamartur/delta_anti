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

# Risk Management - Conservative for $100 wallet
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.10"))  # 10% risk per trade
RISK_AMOUNT_USD = float(os.getenv("RISK_AMOUNT_USD", "2"))  # Fixed $2 risk per trade (fallback)
MAX_POSITION_SIZE = int(os.getenv("MAX_POSITION_SIZE", "100"))
MAX_DAILY_TRADES = 20  # More trades allowed
MAX_DRAWDOWN_PERCENT = 0.30  # Conservative 30% max drawdown

# Dynamic Position Sizing
DYNAMIC_POSITION_SIZING = True  # Use percentage of balance instead of fixed USD
MIN_TRADE_SIZE_USD = 1.0  # Minimum trade size in USD

# Leverage Configuration
DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "200"))  # 200x leverage per trade
AUTO_TOPUP = os.getenv("AUTO_TOPUP", "true").lower() == "true"  # Auto topup to prevent liquidation

# Margin Usage - Use 95% of available balance for maximum profit
MARGIN_USAGE_PCT = float(os.getenv("MARGIN_USAGE_PCT", "0.95"))

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

# Signal Configuration - AGGRESSIVE: LARGER PROFITS, LOWER WIN RATE
SIGNAL_CONFIG = {
    "min_confidence": 80,  # STRICT: Only trade on high-confidence signals (80%+)
    "confirm_candles": 1,   # Quick entry
    # PROFIT TARGET: Aggressive 2:1 R:R ratio
    # TP is 2x the SL - can be wrong 50% of the time and still profit!
    # Let winners run, cut losers quickly
    "atr_multiplier_tp": 1.0,  # Take profit at 1.0x ATR (let winners run)
    "atr_multiplier_sl": 0.5,  # Tight stop loss at 0.5x ATR (cut losers fast)
    "min_profit_target_pct": 1.0,  # Target 1% account return per trade
}

# Adaptive Trading Configuration (learns from trade history)
ADAPTIVE_TRADING_CONFIG = {
    "enabled": True,                    # Enable/disable adaptive learning
    "min_historical_trades": 5,         # Min trades needed to learn from a combo
    "min_win_rate_threshold": 30,       # Block combos with <30% win rate
    "min_confidence_threshold": 80,     # STRICT: Match signal config (80%+)
    "refresh_interval_minutes": 30,     # How often to re-analyze history
}

# Pre-Trade Market Analysis Configuration
MARKET_ANALYSIS_CONFIG = {
    "enabled": True,                    # Enable/disable market analysis
    "min_trade_quality": 60,            # Minimum quality score to trade (0-100)
    "min_trend_confidence": 50,         # Minimum trend confidence for trend trades
    "sr_lookback_periods": 50,          # Periods for support/resistance detection
    "min_rr_ratio": 1.5,                # Minimum risk/reward ratio
    "use_ai_confirmation": True,        # Require AI model agreement
    # Noise filter settings
    "noise_filter": {
        "noise_threshold": 60,          # Block signals above this noise score
        "atr_spike_threshold": 2.0,     # ATR ratio > 2x = volatility spike
        "min_body_ratio": 0.3,          # Candle body must be 30% of range
        "volume_z_threshold": 2.5,      # Volume z-score threshold
        "max_direction_changes": 4,     # Max reversals in last 10 candles
    }
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

# ============ ML Validation Thresholds (Tuned) ============
ML_VALIDATION_CONFIG = {
    "ml_confidence_threshold": 60,      # Require 60%+ ML confidence to act (was 50)
    "sentiment_score_threshold": 0.3,   # Require |score| > 0.3 to act (was 0.2)
    "ml_confirm_boost": 15,             # Confidence boost when ML confirms
    "ml_block_on_disagree": True,       # Block signal when ML strongly disagrees
    "sentiment_confirm_boost": 10,      # Confidence boost when sentiment confirms
    "sentiment_penalty": -10,           # Confidence penalty when sentiment disagrees
}

# ============ Adaptive Strategy Configuration ============
ADAPTIVE_STRATEGY_CONFIG = {
    "enabled": True,
    "regime_lookback": 20,              # Candles to look back for regime detection
    "adx_trend_threshold": 25,          # ADX > 25 = Trending
    "adx_range_threshold": 20,          # ADX < 20 = Ranging
    "bb_squeeze_threshold": 3.0,        # BB Width < 3% = Low Volatility/Range
    "volatility_penalty": 20,           # Confidence penalty in high volatility
}

# ============ Backtesting Configuration ============
BACKTEST_CONFIG = {
    "initial_balance": 10000.0,         # Start with $10,000
    "commission_rate": 0.0006,          # 0.06% taker fee
    "slippage_pct": 0.0002,             # 0.02% slippage assumption
    "use_fixed_size": False,            # If True, use fixed quantity. If False, use adaptive dynamic sizing.
    "fixed_position_size": 1.0,         # 1.0 BTC if fixed
}

# ============ Multi-Timeframe Configuration ============
MULTI_TIMEFRAME_CONFIG = {
    "enabled": True,
    "higher_timeframe": "1h",           # Higher TF for trend confirmation
    "trend_agreement_required": True,   # Only trade when HTF trend aligns
    "htf_lookback_hours": 24,           # Hours of HTF data to load
    "htf_trend_boost": 10,              # Confidence boost when HTF confirms
    "htf_block_penalty": True,          # Block trades against HTF trend
}

# ============ Ensemble Model Voting ============
ENSEMBLE_CONFIG = {
    "enabled": True,
    "require_consensus": False,         # Don't require both to agree
    "consensus_bonus": 20,              # Extra confidence when both models agree
    "single_model_weight": 1.0,         # Weight for single model prediction
    "consensus_weight": 1.5,            # Weight multiplier for consensus predictions
    "min_models_required": 1,           # Minimum models needed for prediction
}

# ============ Dynamic Position Sizing (Confluence) ============
DYNAMIC_SIZING_CONFIG = {
    "enabled": True,
    "base_risk_pct": 0.10,              # Base 10% risk
    "max_risk_pct": 0.25,               # Max 25% risk on high confluence
    "confluence_tiers": {
        # confidence_range: risk_multiplier
        80: 1.0,    # 80-85 confidence → 1x base risk (2%)
        86: 1.5,    # 86-90 confidence → 1.5x risk (3%)
        91: 2.0,    # 91-95 confidence → 2x risk (4%)
        96: 2.5,    # 96-100 confidence → 2.5x risk (5%)
    }
}

# ============ DQN Agent Integration ============
DQN_INTEGRATION_CONFIG = {
    "enabled": True,
    "dqn_confirm_boost": 10,            # Confidence boost when DQN agrees
    "dqn_hold_penalty": -5,             # Penalty when DQN says HOLD
    "dqn_oppose_penalty": -15,          # Penalty when DQN opposes direction
    "train_on_outcomes": True,          # Enable online learning from trade results
    "save_every_n_trades": 10,          # Save DQN model every N completed trades
}

# ============ Model Accuracy Weighting ============
MODEL_ACCURACY_CONFIG = {
    "enabled": True,
    "window_size": 50,              # Rolling window of trades to track
    "min_trades": 10,               # Min trades before adjusting weights
    "base_weight": 1.0,             # Default weight multiplier
    "max_weight": 1.5,              # Max boost for high-accuracy models
    "min_weight": 0.3,              # Min weight for poor-accuracy models
    "save_path": "models/model_accuracy.json",
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
# Sync from exchange on startup, fallback to env or default
INITIAL_ACCOUNT_BALANCE = float(os.getenv("INITIAL_BALANCE", "100.0"))

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
    'max_risk_per_trade': float(os.getenv("MAX_RISK_PER_TRADE", "0.10")),  # 10%
    'max_positions': int(os.getenv("MAX_POSITIONS", "10")),  # Increased to 10
    'max_daily_loss': float(os.getenv("MAX_DAILY_LOSS", "0.50")),  # 50% daily loss limit (raised for testing)
    'max_drawdown': float(os.getenv("MAX_DRAWDOWN", "0.70")),  # 70% max drawdown (from peak)
    'trailing_stop_pct': float(os.getenv("TRAILING_STOP_PCT", "1.2")),  # 1.2% - looser to let winners run
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

