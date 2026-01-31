"""
HFT Configuration
All high-frequency trading parameters
"""
import os

# Load from environment or use defaults
HFT_CONFIG = {
    # Network
    'exchange_url': os.getenv('DELTA_WS_URL', 'wss://api.delta.exchange/v2/ws'),
    'enable_binary_protocol': False,  # Delta uses JSON
    'connection_timeout': 5,
    'max_reconnects': 10,
    
    # Symbols to trade
    'symbols': ['BTCUSD', 'ETHUSD'],
    
    # Market Making
    'enable_market_making': True,
    'mm_spread_bps': 5.0,           # Target spread in basis points
    'mm_quote_size': 1.0,           # Contracts per quote
    'mm_max_inventory': 20.0,       # Max position size
    'mm_update_frequency_ms': 100,  # Quote update frequency
    'mm_skew_factor': 0.3,          # Inventory skew
    'mm_min_edge_bps': 2.0,         # Minimum edge required
    
    # Statistical Arbitrage
    'enable_stat_arb': True,
    'stat_arb_pairs': [('BTCUSD', 'ETHUSD')],
    'stat_arb_z_threshold': 2.0,    # Entry z-score
    'stat_arb_exit_z': 0.5,         # Exit z-score
    'stat_arb_lookback': 100,       # Window size
    
    # Risk Management
    'max_order_rate_per_second': 100,
    'max_position_value_usd': 100000,
    'max_daily_loss_usd': 1000,
    'kill_switch_loss_pct': 5.0,
    
    # Performance
    'order_book_depth': 100,
    'latency_window_size': 10000,
    'enable_performance_profiling': True,
    
    # Monitoring
    'latency_alert_threshold_ms': 10,
    'print_stats_interval_sec': 30,
    'log_level': 'INFO'
}

# Risk controls (critical)
HFT_RISK_CONTROLS = {
    # Order Rate Limiting
    'max_orders_per_second': 100,
    'max_cancels_per_second': 200,
    
    # Position Limits
    'max_position_size': 20.0,
    'max_position_value_usd': 100000,
    
    # Loss Limits
    'max_loss_per_trade_usd': 100,
    'daily_loss_limit_usd': 1000,
    'kill_switch_threshold_usd': 5000,
    
    # Latency Limits
    'max_acceptable_latency_ms': 50,
    'pause_trading_if_slow': True
}
