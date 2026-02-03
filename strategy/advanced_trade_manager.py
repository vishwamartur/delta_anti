"""
Advanced Trade Manager for Delta Exchange
Handles position management, order execution, risk controls
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import uuid

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.delta_rest import rest_client
from analysis.indicators import IndicatorValues
import config


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradeState(Enum):
    """Trade lifecycle states"""
    PENDING = "PENDING"           # Signal generated, not executed
    CREATED = "CREATED"           # Order placed, waiting fill
    ACTIVE = "ACTIVE"             # Position open
    CLOSING = "CLOSING"           # Exit order placed
    CLOSED = "CLOSED"             # Position closed
    CANCELLED = "CANCELLED"       # Order cancelled
    REJECTED = "REJECTED"         # Order rejected


class TradeDirection(Enum):
    """Trade direction"""
    LONG = "LONG"
    SHORT = "SHORT"


class ExitReason(Enum):
    """Exit reasons for analytics"""
    TAKE_PROFIT = "Take Profit"
    STOP_LOSS = "Stop Loss"
    TRAILING_STOP = "Trailing Stop"
    MANUAL = "Manual Close"
    SIGNAL_EXIT = "Signal Exit"
    TIME_EXIT = "Time Exit"
    RISK_LIMIT = "Risk Limit Breach"
    DAILY_LOSS_LIMIT = "Daily Loss Limit"


@dataclass
class Order:
    """Order details"""
    order_id: Optional[str] = None
    client_order_id: str = ""
    symbol: str = ""
    side: str = ""  # buy/sell
    order_type: str = "limit"  # market/limit
    size: float = 0.0
    price: Optional[float] = None
    status: str = "pending"  # pending/open/filled/cancelled/rejected
    filled_size: float = 0.0
    average_fill_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'size': self.size,
            'price': self.price,
            'status': self.status,
            'filled_size': self.filled_size,
            'average_fill_price': self.average_fill_price
        }


@dataclass
class Trade:
    """Trade position with full lifecycle tracking"""
    # Identification
    trade_id: str
    symbol: str
    direction: TradeDirection
    
    # Entry details
    entry_price: Optional[float] = None
    entry_size: float = 0.0
    entry_time: Optional[datetime] = None
    entry_order: Optional[Order] = None
    
    # Exit details
    exit_price: Optional[float] = None
    exit_size: float = 0.0
    exit_time: Optional[datetime] = None
    exit_order: Optional[Order] = None
    exit_reason: Optional[ExitReason] = None
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pct: float = 0.0
    trailing_stop_price: Optional[float] = None
    
    # State tracking
    state: TradeState = TradeState.PENDING
    current_price: float = 0.0
    
    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    pnl_percent: float = 0.0
    fees_paid: float = 0.0
    
    # ML metadata
    ml_confidence: float = 0.0
    ml_prediction: Optional[float] = None
    entry_indicators: Optional[Dict] = None
    
    # Strategy metadata
    strategy_name: str = ""
    timeframe: str = ""
    notes: str = ""
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_long(self) -> bool:
        return self.direction == TradeDirection.LONG
    
    @property
    def is_short(self) -> bool:
        return self.direction == TradeDirection.SHORT
    
    @property
    def duration_minutes(self) -> int:
        if self.entry_time:
            end_time = self.exit_time or datetime.now()
            return int((end_time - self.entry_time).total_seconds() / 60)
        return 0
    
    def update_pnl(self, current_price: float, total_fee_pct: float = 0.12):
        """Update unrealized P&L with fee deduction"""
        self.current_price = current_price
        self.updated_at = datetime.now()
        
        if self.entry_price and self.entry_size > 0:
            if self.is_long:
                pnl_per_unit = current_price - self.entry_price
            else:
                pnl_per_unit = self.entry_price - current_price
            
            # Gross P&L
            gross_pnl = pnl_per_unit * self.entry_size
            
            # Calculate fees (spread + taker for entry and exit)
            trade_value = self.entry_price * self.entry_size
            estimated_fees = trade_value * (total_fee_pct / 100) * 2  # Entry + exit
            
            # Net P&L after fees
            self.unrealized_pnl = gross_pnl - estimated_fees
            self.fees_paid = estimated_fees
            self.pnl_percent = ((gross_pnl - estimated_fees) / trade_value) * 100
    
    def calculate_realized_pnl(self, total_fee_pct: float = 0.12):
        """Calculate realized P&L after exit with fees"""
        if self.entry_price and self.exit_price and self.exit_size > 0:
            if self.is_long:
                pnl_per_unit = self.exit_price - self.entry_price
            else:
                pnl_per_unit = self.entry_price - self.exit_price
            
            # Gross P&L
            gross_pnl = pnl_per_unit * self.exit_size
            
            # Calculate actual fees (spread + taker for entry and exit)
            entry_value = self.entry_price * self.exit_size
            exit_value = self.exit_price * self.exit_size
            
            # Fees on both entry and exit
            entry_fees = entry_value * (total_fee_pct / 100)
            exit_fees = exit_value * (total_fee_pct / 100)
            total_fees = entry_fees + exit_fees
            
            self.fees_paid = total_fees
            self.realized_pnl = gross_pnl - total_fees
            self.pnl_percent = (self.realized_pnl / entry_value) * 100
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'entry_price': self.entry_price,
            'entry_size': self.entry_size,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_price': self.exit_price,
            'exit_size': self.exit_size,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_reason': self.exit_reason.value if self.exit_reason else None,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop_price': self.trailing_stop_price,
            'state': self.state.value,
            'current_price': self.current_price,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'pnl_percent': self.pnl_percent,
            'ml_confidence': self.ml_confidence,
            'strategy_name': self.strategy_name,
            'duration_minutes': self.duration_minutes
        }


class RiskManager:
    """Risk management and position sizing"""
    
    def __init__(self, account_balance: float, risk_config: Dict):
        self.account_balance = account_balance
        self.max_risk_per_trade = risk_config.get('max_risk_per_trade', 0.02)  # 2%
        self.max_positions = risk_config.get('max_positions', 5)
        self.max_daily_loss = risk_config.get('max_daily_loss', 0.05)  # 5%
        self.max_drawdown = risk_config.get('max_drawdown', 0.15)  # 15%
        
        # Tracking
        self.daily_pnl = 0.0
        self.daily_start_balance = account_balance
        self.peak_balance = account_balance
        self.current_drawdown = 0.0
        
        logger.info(f"RiskManager initialized: Balance=${account_balance:,.2f}, "
                   f"MaxRisk={self.max_risk_per_trade*100}%, MaxPos={self.max_positions}")
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                               symbol: str = "BTCUSD") -> float:
        """
        Calculate position size based on fixed USD risk amount.
        Uses RISK_AMOUNT_USD from config ($300 default).
        Position Size = Risk Amount / |Entry - Stop Loss|
        """
        if stop_loss == 0 or entry_price == 0:
            logger.warning("Invalid prices for position sizing")
            return 0.0
        
        # Use fixed USD risk amount from config (default $300)
        risk_amount = getattr(config, 'RISK_AMOUNT_USD', 500)
        
        # Cap at available balance
        risk_amount = min(risk_amount, self.account_balance * 0.95)
        
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0.0
        
        position_size = risk_amount / price_risk
        
        # Apply maximum position limits
        max_position_value = self.account_balance * 0.95  # Use 95% of balance max
        max_size = max_position_value / entry_price * config.DEFAULT_LEVERAGE
        
        position_size = min(position_size, max_size)
        
        logger.info(f"Position sizing: Entry=${entry_price}, SL=${stop_loss}, "
                   f"Risk=${risk_amount:.2f}, Size={position_size:.4f}")
        
        return position_size
    
    def can_open_position(self, open_positions_count: int) -> Tuple[bool, str]:
        """Check if new position can be opened"""
        
        # Check position limit
        if open_positions_count >= self.max_positions:
            return False, f"Max positions limit reached ({self.max_positions})"
        
        # Check daily loss limit
        if self.daily_start_balance > 0:
            daily_loss_pct = (self.daily_pnl / self.daily_start_balance) * 100
            if daily_loss_pct <= -self.max_daily_loss * 100:
                return False, f"Daily loss limit reached ({daily_loss_pct:.2f}%)"
        
        # Check drawdown
        if self.peak_balance > 0:
            self.current_drawdown = ((self.peak_balance - self.account_balance) / 
                                     self.peak_balance) * 100
            if self.current_drawdown >= self.max_drawdown * 100:
                return False, f"Max drawdown reached ({self.current_drawdown:.2f}%)"
        
        return True, "OK"
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        self.daily_pnl += pnl
        self.account_balance += pnl
        
        # Update peak for drawdown calculation
        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at start of trading day)"""
        self.daily_pnl = 0.0
        self.daily_start_balance = self.account_balance
        logger.info(f"Daily stats reset. Starting balance: ${self.account_balance:,.2f}")


class AdvancedTradeManager:
    """
    Advanced Trade Manager with order execution, risk management, and P&L tracking
    """
    
    def __init__(self, account_balance: float, trade_config: Dict):
        self.trades: Dict[str, Trade] = {}  # All trades
        self.open_trades: Dict[str, Trade] = {}  # Active positions only
        self.trade_history: List[Trade] = []
        
        # Risk management
        self.risk_manager = RiskManager(account_balance, trade_config)
        
        # Configuration
        self.config = trade_config
        self.enable_auto_execution = trade_config.get('enable_auto_execution', False)
        self.enable_trailing_stop = trade_config.get('enable_trailing_stop', True)
        self.trailing_stop_pct = trade_config.get('trailing_stop_pct', 1.5)
        
        # Fee structure from config (Delta Exchange X-Mas Offer rates)
        # IMPORTANT: Fees are on NOTIONAL value (price × quantity)
        fee_config = getattr(config, 'FEE_CONFIG', {})
        self.maker_fee_pct = fee_config.get('futures_maker', 0.0002) * 100   # 0.02%
        self.taker_fee_pct = fee_config.get('futures_taker', 0.0005) * 100   # 0.05%
        self.total_fee_pct = self.taker_fee_pct * 2  # Round-trip (entry + exit)
        self.min_profit_pct = trade_config.get('min_profit_pct', self.total_fee_pct + 0.02)  # Min profit after fees
        
        # Product ID mapping
        self.product_ids = {}
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees_paid = 0.0
        
        # Persistence
        self.trades_file = Path("data/trades.json")
        self.trades_file.parent.mkdir(exist_ok=True)
        
        self._load_trades()
        self._load_product_ids()
        
        # Sync balance from Delta Exchange on startup
        if trade_config.get('sync_balance_on_start', True):
            self._sync_balance_on_start()
        
        # Setup leverage and auto-topup for all trading symbols
        if self.enable_auto_execution:
            self._setup_leverage_and_topup()
        
        logger.info(f"AdvancedTradeManager initialized: auto_execution={self.enable_auto_execution}, "
                   f"balance=${self.risk_manager.account_balance:,.2f}, "
                   f"fees={self.total_fee_pct}%")
        
        # Trade frequency tracking (to reduce overtrading)
        self._last_trade_time = None
    
    def _sync_balance_on_start(self):
        """Sync account balance from Delta Exchange on startup"""
        try:
            response = rest_client.get_wallet_balances()
            
            if 'result' not in response:
                logger.warning(f"[SYNC] Could not sync balance: {response}")
                return
            
            for wallet in response['result']:
                asset = wallet.get('asset_symbol', wallet.get('asset', ''))
                balance = float(wallet.get('balance', 0))
                
                # Use USD or USDT balance
                if asset in ('USD', 'USDT', 'INR'):
                    self.risk_manager.account_balance = balance
                    self.risk_manager.peak_balance = balance
                    self.risk_manager.daily_start_balance = balance
                    logger.info(f"[SYNC] Balance synced from Delta: ${balance:,.2f} {asset}")
                    return
            
            logger.warning("[SYNC] No USD/USDT balance found on Delta Exchange")
            
        except Exception as e:
            logger.error(f"[SYNC] Failed to sync balance: {e}")
    
    def _load_product_ids(self):
        """Load product IDs from Delta Exchange"""
        try:
            products = rest_client.get_products()
            if 'result' in products:
                for p in products['result']:
                    symbol = p.get('symbol', '')
                    product_id = p.get('id')
                    if symbol and product_id:
                        self.product_ids[symbol] = product_id
                logger.info(f"Loaded {len(self.product_ids)} product IDs")
        except Exception as e:
            logger.warning(f"Could not load product IDs: {e}")
            # Fallback mapping
            self.product_ids = {"BTCUSD": 139, "ETHUSD": 132}
    
    def _setup_leverage_and_topup(self):
        """Setup 200x leverage and auto-topup for all trading symbols"""
        import config
        
        logger.info(f"[SETUP] Configuring {config.DEFAULT_LEVERAGE}x leverage and auto-topup...")
        
        for symbol in config.TRADING_SYMBOLS:
            product_id = self.product_ids.get(symbol)
            if not product_id:
                logger.warning(f"[SETUP] No product ID for {symbol}, skipping")
                continue
            
            try:
                # Set leverage
                leverage_result = rest_client.set_leverage(product_id, config.DEFAULT_LEVERAGE)
                if leverage_result.get('success') or leverage_result.get('result'):
                    logger.info(f"[LEVERAGE] Set {config.DEFAULT_LEVERAGE}x leverage for {symbol}")
                else:
                    logger.warning(f"[LEVERAGE] Failed to set leverage for {symbol}: {leverage_result}")
                
                # Enable auto-topup (if AUTO_TOPUP is enabled in config)
                if config.AUTO_TOPUP:
                    topup_result = rest_client.set_auto_topup(product_id, enabled=True)
                    if topup_result.get('success') or topup_result.get('result'):
                        logger.info(f"[AUTO-TOPUP] Enabled for {symbol}")
                    else:
                        # Auto-topup may fail if no position exists, which is fine
                        logger.debug(f"[AUTO-TOPUP] Note for {symbol}: {topup_result}")
                        
            except Exception as e:
                logger.error(f"[SETUP] Error configuring {symbol}: {e}")
    
    def sync_positions(self) -> Dict:
        """
        Sync internal state with actual Delta Exchange positions.
        Returns current positions from exchange.
        """
        try:
            response = rest_client.get_positions()
            
            if 'result' not in response:
                logger.warning(f"[SYNC] No positions data: {response}")
                return {}
            
            positions = response['result']
            live_positions = {}
            
            for pos in positions:
                symbol = pos.get('product', {}).get('symbol', pos.get('product_symbol', ''))
                size = float(pos.get('size', 0))
                
                if size == 0:
                    continue
                
                entry_price = float(pos.get('entry_price', 0))
                mark_price = float(pos.get('mark_price', 0))
                unrealized_pnl = float(pos.get('unrealized_pnl', 0))
                realized_pnl = float(pos.get('realized_pnl', 0))
                margin = float(pos.get('margin', 0))
                liquidation_price = float(pos.get('liquidation_price', 0) or 0)
                
                direction = 'LONG' if size > 0 else 'SHORT'
                
                live_positions[symbol] = {
                    'symbol': symbol,
                    'size': abs(size),
                    'direction': direction,
                    'entry_price': entry_price,
                    'mark_price': mark_price,
                    'unrealized_pnl': unrealized_pnl,
                    'realized_pnl': realized_pnl,
                    'margin': margin,
                    'liquidation_price': liquidation_price,
                    'pnl_percent': (unrealized_pnl / margin * 100) if margin > 0 else 0
                }
                
                logger.info(f"[SYNC] {symbol}: {direction} x{abs(size):.4f} @ ${entry_price:.2f}, "
                           f"P&L: ${unrealized_pnl:.2f} ({live_positions[symbol]['pnl_percent']:.2f}%)")
            
            # Sync with internal trades
            self._sync_with_live_positions(live_positions)
            
            return live_positions
            
        except Exception as e:
            logger.error(f"[SYNC] Failed to sync positions: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _sync_with_live_positions(self, live_positions: Dict):
        """Sync internal trade tracking with live positions"""
        
        # Update existing trades with live data
        for trade_id, trade in list(self.open_trades.items()):
            if trade.symbol in live_positions:
                pos = live_positions[trade.symbol]
                trade.current_price = pos['mark_price']
                trade.unrealized_pnl = pos['unrealized_pnl']
                trade.pnl_percent = pos['pnl_percent']
                trade.updated_at = datetime.now()
            else:
                # Position closed externally
                logger.warning(f"[SYNC] Trade {trade_id} not found on exchange - marking closed")
                trade.state = TradeState.CLOSED
                trade.exit_reason = ExitReason.MANUAL
                trade.exit_time = datetime.now()
                del self.open_trades[trade_id]
                self.trade_history.insert(0, trade)
        
        # Check for positions not in our tracking
        for symbol, pos in live_positions.items():
            has_trade = any(t.symbol == symbol for t in self.open_trades.values())
            if not has_trade:
                logger.info(f"[SYNC] Found untracked position: {symbol} - creating trade")
                # Create trade from live position
                trade_id = f"EXT{int(datetime.now().timestamp() * 1000)}"
                direction = TradeDirection.LONG if pos['direction'] == 'LONG' else TradeDirection.SHORT
                
                trade = Trade(
                    trade_id=trade_id,
                    symbol=symbol,
                    direction=direction,
                    entry_price=pos['entry_price'],
                    entry_size=pos['size'],
                    entry_time=datetime.now(),
                    current_price=pos['mark_price'],
                    unrealized_pnl=pos['unrealized_pnl'],
                    pnl_percent=pos['pnl_percent'],
                    state=TradeState.ACTIVE,
                    strategy_name='external',
                    notes='Synced from exchange'
                )
                
                self.trades[trade_id] = trade
                self.open_trades[trade_id] = trade
        
        self._save_trades()
    
    def sync_balance(self) -> Dict:
        """
        Sync account balance from Delta Exchange.
        Returns wallet balances.
        """
        try:
            response = rest_client.get_wallet_balances()
            
            if 'result' not in response:
                logger.warning(f"[SYNC] No balance data: {response}")
                return {}
            
            balances = {}
            for wallet in response['result']:
                asset = wallet.get('asset_symbol', wallet.get('asset', 'USD'))
                balance = float(wallet.get('balance', 0))
                available = float(wallet.get('available_balance', balance))
                
                balances[asset] = {
                    'total': balance,
                    'available': available,
                    'margin_used': balance - available
                }
                
                # Update risk manager if USD
                if asset == 'USD' or asset == 'USDT':
                    self.risk_manager.account_balance = balance
                    if balance > self.risk_manager.peak_balance:
                        self.risk_manager.peak_balance = balance
            
            logger.info(f"[SYNC] Balance synced: {balances}")
            return balances
            
        except Exception as e:
            logger.error(f"[SYNC] Failed to sync balance: {e}")
            return {}
    
    def get_live_positions(self) -> List[Dict]:
        """Get list of current live positions from exchange"""
        positions = self.sync_positions()
        return list(positions.values())
    
    def get_account_summary(self) -> Dict:
        """Get full account summary with positions and balance"""
        balances = self.sync_balance()
        positions = self.sync_positions()
        
        total_unrealized = sum(p.get('unrealized_pnl', 0) for p in positions.values())
        total_margin = sum(p.get('margin', 0) for p in positions.values())
        
        return {
            'balances': balances,
            'positions': list(positions.values()),
            'position_count': len(positions),
            'total_unrealized_pnl': total_unrealized,
            'total_margin_used': total_margin,
            'open_trades': len(self.open_trades),
            'daily_pnl': self.risk_manager.daily_pnl,
            'account_balance': self.risk_manager.account_balance
        }
    
    def check_market_conditions(self, symbol: str, indicators: IndicatorValues, 
                                 direction: str) -> Tuple[bool, str]:
        """
        Pre-entry market condition check.
        
        Validates market is suitable for trading before entering position.
        
        Checks:
        1. Volatility (ADX) - Avoid dead/choppy markets
        2. Spread health - Avoid wide spreads
        3. RSI extremes - Don't enter at exhaustion
        4. Trend clarity - Ensure clear direction
        
        Args:
            symbol: Trading symbol
            indicators: Current indicator values
            direction: 'LONG' or 'SHORT'
            
        Returns:
            Tuple of (is_suitable, reason)
        """
        reasons = []
        
        # 1. VOLATILITY CHECK (ADX)
        # ADX < 15 = dead/choppy market, not suitable for directional trades
        # ADX > 50 = extreme volatility, higher risk
        if indicators.adx is not None:
            if indicators.adx < 15:
                return False, f"Market too quiet (ADX={indicators.adx:.1f}<15) - wait for volatility"
            if indicators.adx > 50:
                reasons.append(f"High volatility (ADX={indicators.adx:.1f})")
        
        # 2. SPREAD CHECK (via BB width as proxy)
        # Very tight BB = low volatility, potential squeeze incoming
        bb_width = 0
        if indicators.bb_upper and indicators.bb_lower and indicators.price > 0:
            bb_width = ((indicators.bb_upper - indicators.bb_lower) / indicators.price) * 100
            if bb_width < 0.3:  # Less than 0.3% width = very tight
                return False, f"Bollinger squeeze (width={bb_width:.2f}%) - wait for breakout"
        
        # 3. RSI EXTREME CHECK - Don't enter into exhaustion
        if indicators.rsi is not None:
            if direction == 'LONG' and indicators.rsi > 75:
                return False, f"RSI overbought ({indicators.rsi:.0f}>75) - avoid long entry"
            if direction == 'SHORT' and indicators.rsi < 25:
                return False, f"RSI oversold ({indicators.rsi:.0f}<25) - avoid short entry"
        
        # 4. TREND CLARITY CHECK - Ensure EMA alignment matches direction
        if indicators.ema_9 and indicators.ema_21:
            ema_bullish = indicators.ema_9 > indicators.ema_21
            if direction == 'LONG' and not ema_bullish:
                reasons.append("EMA bearish (counter-trend trade)")
            elif direction == 'SHORT' and ema_bullish:
                reasons.append("EMA bullish (counter-trend trade)")
        
        # 5. MACD CONFIRMATION - Check for divergence
        if indicators.macd_histogram is not None:
            if direction == 'LONG' and indicators.macd_histogram < 0:
                reasons.append("MACD histogram negative")
            elif direction == 'SHORT' and indicators.macd_histogram > 0:
                reasons.append("MACD histogram positive")
        
        # Allow trade with warnings if not blocked
        if reasons:
            logger.info(f"[MARKET] {symbol} {direction}: Warnings - {', '.join(reasons)}")
        else:
            logger.info(f"[MARKET] {symbol} {direction}: Conditions favorable (ADX={indicators.adx:.1f})")
        
        return True, "Market conditions acceptable"
    
    def create_trade_from_signal(self, signal, indicators: IndicatorValues) -> Optional[Trade]:
        """Create trade from signal generator output"""
        
        # Check risk limits
        can_trade, reason = self.risk_manager.can_open_position(len(self.open_trades))
        if not can_trade:
            logger.warning(f"Cannot open trade: {reason}")
            return None
        
        # Check trade frequency cooldown (to reduce fees from overtrading)
        min_interval = getattr(config, 'MIN_TRADE_INTERVAL_SECONDS', 60)
        if hasattr(self, '_last_trade_time') and self._last_trade_time:
            elapsed = (datetime.now() - self._last_trade_time).total_seconds()
            if elapsed < min_interval:
                remaining = min_interval - elapsed
                logger.info(f"[COOLDOWN] Trade frequency limit - wait {remaining:.0f}s")
                return None
        
        # Determine direction early for market check
        signal_type = signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type)
        direction = TradeDirection.LONG if 'LONG' in signal_type.upper() else TradeDirection.SHORT
        
        # CHECK MARKET CONDITIONS before entering
        market_ok, market_reason = self.check_market_conditions(
            signal.symbol, indicators, direction.value
        )
        if not market_ok:
            logger.warning(f"[MARKET] Trade blocked: {market_reason}")
            return None
        
        # Generate trade ID
        trade_id = f"T{int(datetime.now().timestamp() * 1000)}"
        
        # Get entry price and ATR
        entry_price = signal.entry_price
        atr = indicators.atr if indicators.atr else entry_price * 0.01  # 1% fallback
        
        # Calculate stop loss and take profit
        if direction == TradeDirection.LONG:
            stop_loss = signal.stop_loss if signal.stop_loss else (entry_price - 2 * atr)
            take_profit = signal.take_profit if signal.take_profit else (entry_price + 3 * atr)
            
            # VALIDATION: For LONG, SL must be below entry, TP must be above entry
            if stop_loss >= entry_price:
                logger.warning(f"[VALIDATION] LONG SL ${stop_loss:.2f} >= entry ${entry_price:.2f}, fixing...")
                stop_loss = entry_price - 2 * atr
            if take_profit <= entry_price:
                logger.warning(f"[VALIDATION] LONG TP ${take_profit:.2f} <= entry ${entry_price:.2f}, fixing...")
                take_profit = entry_price + 3 * atr
        else:
            stop_loss = signal.stop_loss if signal.stop_loss else (entry_price + 2 * atr)
            take_profit = signal.take_profit if signal.take_profit else (entry_price - 3 * atr)
            
            # VALIDATION: For SHORT, SL must be above entry, TP must be below entry
            if stop_loss <= entry_price:
                logger.warning(f"[VALIDATION] SHORT SL ${stop_loss:.2f} <= entry ${entry_price:.2f}, fixing...")
                stop_loss = entry_price + 2 * atr
            if take_profit >= entry_price:
                logger.warning(f"[VALIDATION] SHORT TP ${take_profit:.2f} >= entry ${entry_price:.2f}, fixing...")
                take_profit = entry_price - 3 * atr
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            entry_price, stop_loss, signal.symbol
        )
        
        if position_size <= 0:
            logger.warning("Position size calculated as 0")
            return None
        
        # Create trade object
        trade = Trade(
            trade_id=trade_id,
            symbol=signal.symbol,
            direction=direction,
            entry_price=entry_price,
            entry_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=self.trailing_stop_pct,
            ml_confidence=getattr(signal, 'confidence', 0),
            entry_indicators={
                'rsi': indicators.rsi,
                'macd': indicators.macd_line,
                'atr': indicators.atr
            },
            strategy_name=self.config.get('strategy_name', 'delta_anti'),
            state=TradeState.PENDING
        )
        
        self.trades[trade_id] = trade
        
        logger.info(f"Trade created: {trade_id} - {direction.value} {signal.symbol} "
                   f"@ ${entry_price:.2f}, Size={position_size:.4f}, "
                   f"SL=${stop_loss:.2f}, TP=${take_profit:.2f}")
        
        # Auto-execute if enabled
        if self.enable_auto_execution:
            self.execute_trade(trade)
        else:
            # Paper trading - simulate immediate fill
            trade.entry_time = datetime.now()
            trade.state = TradeState.ACTIVE
            self.open_trades[trade_id] = trade
            
            # ===== DETAILED TRADE ENTRY LOG =====
            notional = trade.entry_price * trade.entry_size
            sl_distance = abs(trade.entry_price - trade.stop_loss)
            tp_distance = abs(trade.take_profit - trade.entry_price)
            sl_pct = (sl_distance / trade.entry_price * 100) if trade.entry_price else 0
            tp_pct = (tp_distance / trade.entry_price * 100) if trade.entry_price else 0
            est_fees = notional * (self.total_fee_pct / 100)
            rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
            
            logger.info("=" * 60)
            logger.info(f"[ENTRY] {trade.symbol} {trade.direction.value} @ ${trade.entry_price:,.2f}")
            logger.info("-" * 40)
            logger.info(f"  Size:      {trade.entry_size} contracts")
            logger.info(f"  Notional:  ${notional:,.2f}")
            logger.info(f"  Stop Loss: ${trade.stop_loss:,.2f} ({sl_pct:.2f}% away)")
            logger.info(f"  Take Profit: ${trade.take_profit:,.2f} ({tp_pct:.2f}% away)")
            logger.info(f"  Risk/Reward: {rr_ratio:.1f}x")
            logger.info(f"  Est. Fees:   ${est_fees:.2f} (round-trip)")
            logger.info("=" * 60)
        
        # Update last trade time for cooldown tracking
        self._last_trade_time = datetime.now()
        
        self._save_trades()
        return trade
    
    def execute_trade(self, trade: Trade) -> bool:
        """Execute trade by placing order via Delta Exchange API"""
        
        try:
            # Determine order side
            side = "buy" if trade.is_long else "sell"
            product_id = self.product_ids.get(trade.symbol, 139)
            
            # Ensure size is at least 1 contract
            order_size = max(1, int(trade.entry_size))
            
            logger.info(f"[LIVE] Placing {side.upper()} LIMIT order: {trade.symbol} x{order_size}")
            
            # Use LIMIT orders to get maker fees (0.02%) instead of taker fees (0.05%)
            # Price slightly better than current to ensure fill while getting maker rebate
            # For BUY: slightly above current price (willing to pay a bit more)
            # For SELL: slightly below current price (willing to accept a bit less)
            order_type = "limit_order"
            limit_price = trade.entry_price
            
            # Aggressive limit: 0.01% favorable to ensure quick fill
            price_offset = trade.entry_price * 0.0001  # 0.01%
            if side == "buy":
                limit_price = trade.entry_price + price_offset  # Slightly higher buy limit
            else:
                limit_price = trade.entry_price - price_offset  # Slightly lower sell limit
            
            response = rest_client.place_order(
                symbol=trade.symbol,
                side=side,
                size=order_size,
                order_type=order_type,
                limit_price=round(limit_price, 2)  # Round to avoid precision issues
            )
            
            logger.info(f"[LIVE] Order response: {response}")
            
            if response.get('success') or response.get('result'):
                order_data = response.get('result', {})
                order_id = order_data.get('id')
                
                # Create entry order record
                trade.entry_order = Order(
                    order_id=str(order_id) if order_id else '',
                    client_order_id=order_data.get('client_order_id', ''),
                    symbol=trade.symbol,
                    side=side,
                    order_type='market',
                    size=order_size,
                    status='pending'  # Will update after verification
                )
                
                # ===== FILL VERIFICATION =====
                fill_verified = False
                fill_price = None
                
                # First check if fill_price is in initial response (immediate fill)
                immediate_fill = order_data.get('fill_price') or order_data.get('average_fill_price')
                if immediate_fill:
                    fill_verified = True
                    fill_price = float(immediate_fill)
                    logger.info(f"[FILL] Immediate fill at ${fill_price:.2f}")
                
                # If not immediately filled, poll for fill status
                if not fill_verified and order_id:
                    logger.info(f"[FILL] Waiting for order {order_id} to fill...")
                    fill_result = rest_client.wait_for_order_fill(order_id, max_retries=5, retry_delay=1.0)
                    
                    if fill_result.get('filled'):
                        fill_verified = True
                        fill_price = fill_result.get('fill_price') or trade.entry_price
                        logger.info(f"[FILL] Order {order_id} verified filled at ${fill_price:.2f}")
                    else:
                        logger.warning(f"[FILL] Order {order_id} not confirmed: {fill_result.get('error', 'Unknown')}")
                
                # Final verification: check position exists on exchange
                if fill_verified:
                    position_check = rest_client.verify_position_exists(trade.symbol)
                    if position_check.get('exists'):
                        logger.info(f"[POSITION] Verified: {position_check['direction']} x{position_check['size']} @ ${position_check['entry_price']:.2f}")
                        # Use actual entry price from position if available
                        if position_check.get('entry_price'):
                            fill_price = position_check['entry_price']
                    else:
                        logger.warning(f"[POSITION] Position not found on exchange after fill!")
                
                # Update trade with fill info
                if fill_verified and fill_price:
                    trade.entry_price = fill_price
                    trade.entry_order.average_fill_price = fill_price
                    trade.entry_order.status = 'filled'
                    
                    # ACTIVATE THE TRADE
                    trade.entry_time = datetime.now()
                    trade.entry_size = order_size
                    trade.state = TradeState.ACTIVE
                    trade.current_price = trade.entry_price
                    
                    # Add to open trades
                    self.open_trades[trade.trade_id] = trade
                    
                    self._save_trades()
                    
                    # ===== DETAILED LIVE TRADE ENTRY LOG =====
                    notional = trade.entry_price * trade.entry_size
                    sl_distance = abs(trade.entry_price - trade.stop_loss) if trade.stop_loss else 0
                    tp_distance = abs(trade.take_profit - trade.entry_price) if trade.take_profit else 0
                    sl_pct = (sl_distance / trade.entry_price * 100) if trade.entry_price else 0
                    tp_pct = (tp_distance / trade.entry_price * 100) if trade.entry_price else 0
                    est_fees = notional * (self.total_fee_pct / 100) * 2  # Round-trip
                    rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
                    
                    logger.info("=" * 60)
                    logger.info(f"[LIVE ENTRY] {trade.symbol} {trade.direction.value} @ ${trade.entry_price:,.2f}")
                    logger.info(f"  Order ID: {trade.entry_order.order_id}")
                    logger.info("-" * 40)
                    logger.info(f"  Size:      {trade.entry_size} contracts")
                    logger.info(f"  Notional:  ${notional:,.2f}")
                    logger.info(f"  Stop Loss: ${trade.stop_loss:,.2f} ({sl_pct:.2f}% away)")
                    logger.info(f"  Take Profit: ${trade.take_profit:,.2f} ({tp_pct:.2f}% away)")
                    logger.info(f"  Risk/Reward: {rr_ratio:.1f}x")
                    logger.info(f"  Est. Fees:   ${est_fees:.2f} (round-trip @ {self.total_fee_pct}%)")
                    logger.info("=" * 60)
                    
                    # === PROTECT POSITION: Add max margin + enable auto-topup ===
                    try:
                        # Enable auto-topup to prevent liquidation
                        topup_result = rest_client.set_auto_topup(product_id, enabled=True)
                        logger.info(f"[PROTECTION] Auto-topup enabled for {trade.symbol}")
                        
                        # Add all available margin to position
                        margin_result = rest_client.ensure_max_margin_on_position(trade.symbol)
                        if margin_result.get('success'):
                            logger.info(f"[PROTECTION] Added ${margin_result.get('margin_added', 0):.2f} margin to {trade.symbol}")
                        else:
                            logger.debug(f"[PROTECTION] Margin add note: {margin_result}")
                    except Exception as protection_error:
                        logger.warning(f"[PROTECTION] Could not fully protect position: {protection_error}")
                    
                    return True
                else:
                    # Fill not verified - mark as failed
                    logger.error(f"[LIVE] Trade fill NOT verified, marking as rejected")
                    trade.state = TradeState.REJECTED
                    trade.notes = "Order placed but fill not verified"
                    self._save_trades()
                    return False
                    
            else:
                error_msg = response.get('error', response)
                logger.error(f"[LIVE] Order placement FAILED: {error_msg}")
                trade.state = TradeState.REJECTED
                trade.notes = f"Order rejected: {error_msg}"
                self._save_trades()
                return False
                
        except Exception as e:
            logger.error(f"[LIVE] Exception executing trade: {e}")
            import traceback
            traceback.print_exc()
            trade.state = TradeState.REJECTED
            trade.notes = f"Exception: {str(e)}"
            return False
    
    def update_trade_pnl(self, trade_id: str, current_price: float):
        """Update trade P&L with current market price (including fees)"""
        
        trade = self.trades.get(trade_id)
        if not trade or trade.state != TradeState.ACTIVE:
            return
        
        # Pass fee percentage to P&L calculation
        trade.update_pnl(current_price, self.total_fee_pct)
        
        # Check trailing stop
        if self.enable_trailing_stop and trade.trailing_stop_pct > 0:
            self._update_trailing_stop(trade)
    
    def _update_trailing_stop(self, trade: Trade):
        """Update trailing stop loss based on favorable price movement.
        
        DYNAMIC TRAILING: Tightens as profit increases to lock in more gains.
        - 0.3-0.8% profit: 0.8% trail (base)
        - 0.8-1.5% profit: 0.5% trail (tighter)
        - >1.5% profit: 0.3% trail (very tight to lock in)
        
        ONLY activates after trade is at least 0.3% in profit.
        """
        
        # Calculate current profit percentage
        if trade.entry_price <= 0:
            return
            
        if trade.is_long:
            profit_pct = ((trade.current_price - trade.entry_price) / trade.entry_price) * 100
        else:
            profit_pct = ((trade.entry_price - trade.current_price) / trade.entry_price) * 100
        
        # Only activate trailing stop after minimum profit (0.3%)
        min_profit_for_trailing = 0.3
        if profit_pct < min_profit_for_trailing:
            return  # Don't trail until we have some profit
        
        # DYNAMIC TRAIL: Tighten as profit grows (lock in more gains)
        if profit_pct >= 1.5:
            trail_pct = 0.3  # Very tight - lock in most profit
        elif profit_pct >= 0.8:
            trail_pct = 0.5  # Moderate - balance between profit and breathing room
        else:
            trail_pct = trade.trailing_stop_pct  # Use config default (0.8%)
        
        if trade.is_long:
            # For long positions, trail stop up
            new_stop = trade.current_price * (1 - trail_pct / 100)
            if trade.trailing_stop_price is None or new_stop > trade.trailing_stop_price:
                if new_stop > trade.stop_loss:  # Must be better than original SL
                    trade.trailing_stop_price = new_stop
                    logger.info(f"[TRAIL] {trade.trade_id} trailing stop: ${new_stop:.2f} "
                               f"(profit: {profit_pct:.2f}%, trail: {trail_pct}%)")
        else:
            # For short positions, trail stop down
            new_stop = trade.current_price * (1 + trail_pct / 100)
            if trade.trailing_stop_price is None or new_stop < trade.trailing_stop_price:
                if new_stop < trade.stop_loss:  # Must be better than original SL
                    trade.trailing_stop_price = new_stop
                    logger.info(f"[TRAIL] {trade.trade_id} trailing stop: ${new_stop:.2f} "
                               f"(profit: {profit_pct:.2f}%, trail: {trail_pct}%)")
    
    def check_exit_conditions(self, trade_id: str) -> Optional[ExitReason]:
        """Check if trade should be exited.
        
        Exit checks in priority order:
        1. Trailing stop (lock in profits)
        2. Regular stop loss
        3. Take profit
        4. Time-based exit (stagnant trades)
        """
        
        trade = self.trades.get(trade_id)
        if not trade or trade.state != TradeState.ACTIVE:
            return None
        
        current_price = trade.current_price
        if current_price <= 0:
            return None
        
        # Check trailing stop first (if active)
        if trade.trailing_stop_price:
            if trade.is_long and current_price <= trade.trailing_stop_price:
                return ExitReason.TRAILING_STOP
            elif trade.is_short and current_price >= trade.trailing_stop_price:
                return ExitReason.TRAILING_STOP
        
        # Check regular stop loss
        if trade.stop_loss:
            if trade.is_long and current_price <= trade.stop_loss:
                return ExitReason.STOP_LOSS
            elif trade.is_short and current_price >= trade.stop_loss:
                return ExitReason.STOP_LOSS
        
        # Check take profit
        if trade.take_profit:
            if trade.is_long and current_price >= trade.take_profit:
                return ExitReason.TAKE_PROFIT
            elif trade.is_short and current_price <= trade.take_profit:
                return ExitReason.TAKE_PROFIT
        
        # TIME-BASED EXIT: Close stagnant trades to free capital
        # Exit if trade hasn't reached 0.3% profit within 5 minutes
        if trade.entry_time and trade.entry_price > 0:
            duration_min = trade.duration_minutes
            
            # Calculate current profit %
            if trade.is_long:
                profit_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
            else:
                profit_pct = ((trade.entry_price - current_price) / trade.entry_price) * 100
            
            # Stagnant trade: > 5 min and < 0.3% profit (or slightly negative)
            if duration_min >= 5 and profit_pct < 0.3 and profit_pct > -0.2:
                logger.info(f"[TIME_EXIT] {trade.trade_id} stagnant for {duration_min}m "
                           f"with only {profit_pct:.2f}% profit - freeing capital")
                return ExitReason.TIME_EXIT
        
        return None
    
    def close_trade(self, trade_id: str, exit_reason: ExitReason, 
                   exit_price: Optional[float] = None) -> Optional[Trade]:
        """Close trade and calculate P&L"""
        
        trade = self.trades.get(trade_id)
        if not trade:
            logger.warning(f"Trade {trade_id} not found")
            return None
        
        if trade.state != TradeState.ACTIVE:
            logger.warning(f"Trade {trade_id} not active, state={trade.state.value}")
            return None
        
        # Use current price if not provided
        if exit_price is None:
            exit_price = trade.current_price
        
        # Place exit order if auto-execution is enabled
        if self.enable_auto_execution:
            success = self._place_exit_order(trade, exit_price)
            if not success:
                logger.error(f"Failed to place exit order for {trade_id}")
                # Continue anyway for paper trading
        
        # Update trade
        trade.exit_price = exit_price
        trade.exit_size = trade.entry_size
        trade.exit_time = datetime.now()
        trade.exit_reason = exit_reason
        trade.state = TradeState.CLOSED
        
        # Calculate realized P&L with fees
        trade.calculate_realized_pnl(self.total_fee_pct)
        
        # Track total fees
        self.total_fees_paid += trade.fees_paid
        
        # Update risk manager
        self.risk_manager.update_daily_pnl(trade.realized_pnl)
        
        # Update statistics
        self.total_trades += 1
        if trade.realized_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Move to history
        if trade_id in self.open_trades:
            del self.open_trades[trade_id]
        self.trade_history.insert(0, trade)
        
        # Keep only last 100 in history
        self.trade_history = self.trade_history[:100]
        
        self._save_trades()
        
        # ===== DETAILED TRADE EXIT LOG =====
        duration_sec = (trade.exit_time - trade.entry_time).total_seconds() if trade.entry_time else 0
        price_change = trade.exit_price - trade.entry_price
        price_change_pct = (price_change / trade.entry_price * 100) if trade.entry_price else 0
        
        # Calculate gross P/L before fees
        if trade.is_long:
            gross_pnl = (trade.exit_price - trade.entry_price) * trade.entry_size
        else:
            gross_pnl = (trade.entry_price - trade.exit_price) * trade.entry_size
        
        # Notional value for fee context
        notional = trade.entry_price * trade.entry_size
        
        logger.info("=" * 60)
        logger.info(f"[CLOSED] {trade.symbol} {trade.direction.value} - {exit_reason.value}")
        logger.info("-" * 40)
        logger.info(f"  Entry:     ${trade.entry_price:,.2f} x {trade.entry_size}")
        logger.info(f"  Exit:      ${trade.exit_price:,.2f} ({price_change:+,.2f} / {price_change_pct:+.3f}%)")
        logger.info(f"  Duration:  {int(duration_sec)}s ({trade.duration_minutes}m)")
        logger.info("-" * 40)
        logger.info(f"  Notional:  ${notional:,.2f}")
        logger.info(f"  Gross P/L: ${gross_pnl:+.2f}")
        logger.info(f"  Fees:      ${trade.fees_paid:.2f} ({self.total_fee_pct:.2f}% x2)")
        logger.info(f"  NET P/L:   ${trade.realized_pnl:+.2f} ({trade.pnl_percent:+.2f}%)")
        logger.info("=" * 60)
        
        return trade
    
    def _place_exit_order(self, trade: Trade, exit_price: float) -> bool:
        """Place exit order via API"""
        
        try:
            # Opposite side for exit
            side = "sell" if trade.is_long else "buy"
            order_size = max(1, int(trade.entry_size))
            
            logger.info(f"[LIVE] Placing EXIT {side.upper()} LIMIT order: {trade.symbol} x{order_size}")
            
            # Use LIMIT orders for exits too to get maker fees (0.02% vs 0.05%)
            limit_price = exit_price
            price_offset = exit_price * 0.0001  # 0.01% offset for quick fill
            
            if side == "sell":  # Closing LONG = selling
                limit_price = exit_price - price_offset  # Slightly lower sell limit
            else:  # Closing SHORT = buying
                limit_price = exit_price + price_offset  # Slightly higher buy limit
            
            response = rest_client.place_order(
                symbol=trade.symbol,
                side=side,
                size=order_size,
                order_type="limit_order",
                limit_price=round(limit_price, 2),
                reduce_only=True
            )
            
            logger.info(f"[LIVE] Exit order response: {response}")
            
            if response.get('success') or response.get('result'):
                order_data = response.get('result', {})
                
                trade.exit_order = Order(
                    order_id=str(order_data.get('id', '')),
                    symbol=trade.symbol,
                    side=side,
                    order_type='market',
                    size=order_size,
                    status='filled'
                )
                
                # Get exit fill price
                fill_price = order_data.get('fill_price') or order_data.get('average_fill_price')
                if fill_price:
                    trade.exit_price = float(fill_price)
                    trade.exit_order.average_fill_price = float(fill_price)
                
                logger.info(f"[LIVE] Exit order FILLED: {trade.trade_id}, Price=${trade.exit_price:.2f}")
                return True
            else:
                error_msg = response.get('error', response)
                logger.error(f"[LIVE] Exit order FAILED: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"[LIVE] Exception placing exit order: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def close_all_for_symbol(self, symbol: str, exit_price: float) -> int:
        """Close all positions for a symbol"""
        closed = 0
        for trade_id, trade in list(self.open_trades.items()):
            if trade.symbol == symbol:
                trade.current_price = exit_price
                if self.close_trade(trade_id, ExitReason.MANUAL, exit_price):
                    closed += 1
        return closed
    
    def close_by_direction(self, symbol: str, direction: str, exit_price: float) -> int:
        """Close positions by direction"""
        closed = 0
        target_dir = TradeDirection.LONG if direction.upper() == "LONG" else TradeDirection.SHORT
        for trade_id, trade in list(self.open_trades.items()):
            if trade.symbol == symbol and trade.direction == target_dir:
                trade.current_price = exit_price
                if self.close_trade(trade_id, ExitReason.MANUAL, exit_price):
                    closed += 1
        return closed
    
    def get_open_trade(self, symbol: str) -> Optional[Trade]:
        """Get open trade for symbol"""
        for trade in self.open_trades.values():
            if trade.symbol == symbol:
                return trade
        return None
    
    def has_open_position(self, symbol: str) -> bool:
        """Check if there's an open position for symbol"""
        return self.get_open_trade(symbol) is not None
    
    def sync_trade_stats(self) -> Dict:
        """
        Sync trade statistics from Delta Exchange fills/history.
        Fetches real trades from the exchange.
        """
        try:
            import time
            from datetime import datetime, timedelta
            
            # Get today's start timestamp
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_start_ts = int(today_start.timestamp())
            
            # Fetch fills from Delta Exchange
            response = rest_client.get_fills(page_size=100)
            
            if 'result' not in response:
                logger.warning(f"[SYNC] No fills data: {response}")
                return {}
            
            fills = response.get('result', [])
            
            # Calculate statistics
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            daily_pnl = 0.0
            total_pnl = 0.0
            total_fees = 0.0
            
            # Group fills by order to calculate P&L per trade
            for fill in fills:
                fill_pnl = float(fill.get('realized_pnl', 0))
                fill_fee = float(fill.get('commission', 0))
                fill_time = fill.get('created_at', '')
                
                total_pnl += fill_pnl
                total_fees += fill_fee
                
                # Check if it's a closing fill (has realized P&L)
                if fill_pnl != 0:
                    total_trades += 1
                    net_pnl = fill_pnl - fill_fee
                    
                    if net_pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                    
                    # Check if today's trade
                    try:
                        fill_ts = int(datetime.fromisoformat(fill_time.replace('Z', '+00:00')).timestamp())
                        if fill_ts >= today_start_ts:
                            daily_pnl += net_pnl
                    except:
                        pass
            
            # Update internal counters with exchange data
            if total_trades > 0:
                self.total_trades = total_trades
                self.winning_trades = winning_trades
                self.losing_trades = losing_trades
                self.risk_manager.daily_pnl = daily_pnl
                self.total_fees_paid = total_fees
            
            logger.info(f"[SYNC] Stats from Delta: {total_trades} trades, "
                       f"W:{winning_trades} L:{losing_trades}, Daily P&L: ${daily_pnl:.2f}")
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'total_pnl': total_pnl,
                'daily_pnl': daily_pnl,
                'total_fees': total_fees
            }
            
        except Exception as e:
            logger.error(f"[SYNC] Failed to sync trade stats: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def get_statistics(self) -> Dict:
        """Get trading statistics (syncs from Delta Exchange)"""
        
        # Sync from exchange first
        self.sync_trade_stats()
        
        win_rate = (self.winning_trades / self.total_trades * 100 
                   if self.total_trades > 0 else 0)
        
        total_pnl = sum(t.realized_pnl for t in self.trade_history)
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(win_rate, 1),
            'open_positions': len(self.open_trades),
            'total_pnl': round(total_pnl, 2),
            'daily_pnl': round(self.risk_manager.daily_pnl, 2),
            'account_balance': round(self.risk_manager.account_balance, 2),
            'max_drawdown': round(self.risk_manager.current_drawdown, 2),
            'total_fees_paid': round(self.total_fees_paid, 2)
        }
    
    def _save_trades(self):
        """Save trades to JSON file"""
        try:
            data = {
                'trades': {tid: t.to_dict() for tid, t in self.trades.items()},
                'history': [t.to_dict() for t in self.trade_history[:50]],
                'stats': self.get_statistics(),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.trades_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving trades: {e}")
    
    def _load_trades(self):
        """Load trades from JSON file"""
        try:
            if self.trades_file.exists():
                with open(self.trades_file, 'r') as f:
                    data = json.load(f)
                
                # Restore statistics
                stats = data.get('stats', {})
                self.total_trades = stats.get('total_trades', 0)
                self.winning_trades = stats.get('winning_trades', 0)
                self.losing_trades = stats.get('losing_trades', 0)
                
                logger.info(f"Loaded trade data: {self.total_trades} total trades")
                
        except Exception as e:
            logger.error(f"Error loading trades: {e}")


# Global instance
advanced_trade_manager: Optional[AdvancedTradeManager] = None


def initialize_trade_manager(account_balance: float = None, trade_config: Dict = None) -> AdvancedTradeManager:
    """Initialize the global trade manager"""
    global advanced_trade_manager
    
    if account_balance is None:
        account_balance = getattr(config, 'INITIAL_ACCOUNT_BALANCE', 10000.0)
    
    if trade_config is None:
        trade_config = getattr(config, 'TRADE_MANAGER_CONFIG', {})
    
    advanced_trade_manager = AdvancedTradeManager(account_balance, trade_config)
    return advanced_trade_manager


def get_trade_manager() -> Optional[AdvancedTradeManager]:
    """Get the global trade manager instance"""
    return advanced_trade_manager
