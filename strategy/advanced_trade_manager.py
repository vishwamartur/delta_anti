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
    
    def update_pnl(self, current_price: float):
        """Update unrealized P&L"""
        self.current_price = current_price
        self.updated_at = datetime.now()
        
        if self.entry_price and self.entry_size > 0:
            if self.is_long:
                pnl_per_unit = current_price - self.entry_price
            else:
                pnl_per_unit = self.entry_price - current_price
            
            self.unrealized_pnl = pnl_per_unit * self.entry_size
            self.pnl_percent = (pnl_per_unit / self.entry_price) * 100
    
    def calculate_realized_pnl(self):
        """Calculate realized P&L after exit"""
        if self.entry_price and self.exit_price and self.exit_size > 0:
            if self.is_long:
                pnl_per_unit = self.exit_price - self.entry_price
            else:
                pnl_per_unit = self.entry_price - self.exit_price
            
            self.realized_pnl = (pnl_per_unit * self.exit_size) - self.fees_paid
            self.pnl_percent = (pnl_per_unit / self.entry_price) * 100
    
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
        Calculate position size based on risk percentage
        Position Size = (Account * Risk%) / |Entry - Stop Loss|
        """
        if stop_loss == 0 or entry_price == 0:
            logger.warning("Invalid prices for position sizing")
            return 0.0
        
        risk_amount = self.account_balance * self.max_risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0.0
        
        position_size = risk_amount / price_risk
        
        # Apply maximum position limits (e.g., max 25% of capital per trade)
        max_position_value = self.account_balance * 0.25
        max_size = max_position_value / entry_price
        
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
        
        # Product ID mapping
        self.product_ids = {}
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Persistence
        self.trades_file = Path("data/trades.json")
        self.trades_file.parent.mkdir(exist_ok=True)
        
        self._load_trades()
        self._load_product_ids()
        
        logger.info(f"AdvancedTradeManager initialized: auto_execution={self.enable_auto_execution}")
    
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
    
    def create_trade_from_signal(self, signal, indicators: IndicatorValues) -> Optional[Trade]:
        """Create trade from signal generator output"""
        
        # Check risk limits
        can_trade, reason = self.risk_manager.can_open_position(len(self.open_trades))
        if not can_trade:
            logger.warning(f"Cannot open trade: {reason}")
            return None
        
        # Generate trade ID
        trade_id = f"T{int(datetime.now().timestamp() * 1000)}"
        
        # Determine direction
        signal_type = signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type)
        direction = TradeDirection.LONG if 'LONG' in signal_type.upper() else TradeDirection.SHORT
        
        # Get entry price and ATR
        entry_price = signal.entry_price
        atr = indicators.atr if indicators.atr else entry_price * 0.01  # 1% fallback
        
        # Calculate stop loss and take profit
        if direction == TradeDirection.LONG:
            stop_loss = signal.stop_loss if signal.stop_loss else (entry_price - 2 * atr)
            take_profit = signal.take_profit if signal.take_profit else (entry_price + 3 * atr)
        else:
            stop_loss = signal.stop_loss if signal.stop_loss else (entry_price + 2 * atr)
            take_profit = signal.take_profit if signal.take_profit else (entry_price - 3 * atr)
        
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
            logger.info(f"Paper trade activated: {trade_id}")
        
        self._save_trades()
        return trade
    
    def execute_trade(self, trade: Trade) -> bool:
        """Execute trade by placing order via Delta Exchange API"""
        
        try:
            # Determine order side
            side = "buy" if trade.is_long else "sell"
            product_id = self.product_ids.get(trade.symbol, 139)
            
            # Place market order for entry
            response = rest_client.place_order(
                symbol=trade.symbol,
                side=side,
                size=int(trade.entry_size),
                order_type="market_order"
            )
            
            if response.get('success') or response.get('result'):
                order_data = response.get('result', {})
                
                # Create entry order record
                trade.entry_order = Order(
                    order_id=str(order_data.get('id', '')),
                    client_order_id=order_data.get('client_order_id', ''),
                    symbol=trade.symbol,
                    side=side,
                    order_type='market',
                    size=trade.entry_size,
                    status='open'
                )
                
                trade.state = TradeState.CREATED
                trade.created_at = datetime.now()
                
                self._save_trades()
                
                logger.info(f"Order placed: {trade.trade_id}, OrderID={trade.entry_order.order_id}")
                return True
            else:
                error_msg = response.get('error', 'Unknown error')
                logger.error(f"Order placement failed: {error_msg}")
                trade.state = TradeState.REJECTED
                return False
                
        except Exception as e:
            logger.error(f"Exception executing trade: {e}")
            trade.state = TradeState.REJECTED
            return False
    
    def update_trade_pnl(self, trade_id: str, current_price: float):
        """Update trade P&L with current market price"""
        
        trade = self.trades.get(trade_id)
        if not trade or trade.state != TradeState.ACTIVE:
            return
        
        trade.update_pnl(current_price)
        
        # Check trailing stop
        if self.enable_trailing_stop and trade.trailing_stop_pct > 0:
            self._update_trailing_stop(trade)
    
    def _update_trailing_stop(self, trade: Trade):
        """Update trailing stop loss based on favorable price movement"""
        
        if trade.is_long:
            # For long positions, trail stop up
            new_stop = trade.current_price * (1 - trade.trailing_stop_pct / 100)
            if trade.trailing_stop_price is None or new_stop > trade.trailing_stop_price:
                if trade.trailing_stop_price is None or new_stop > trade.stop_loss:
                    trade.trailing_stop_price = new_stop
                    logger.debug(f"Trailing stop updated: {trade.trade_id} SL=${new_stop:.2f}")
        else:
            # For short positions, trail stop down
            new_stop = trade.current_price * (1 + trade.trailing_stop_pct / 100)
            if trade.trailing_stop_price is None or new_stop < trade.trailing_stop_price:
                if trade.trailing_stop_price is None or new_stop < trade.stop_loss:
                    trade.trailing_stop_price = new_stop
                    logger.debug(f"Trailing stop updated: {trade.trade_id} SL=${new_stop:.2f}")
    
    def check_exit_conditions(self, trade_id: str) -> Optional[ExitReason]:
        """Check if trade should be exited"""
        
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
        
        # Calculate realized P&L
        trade.calculate_realized_pnl()
        
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
        
        logger.info(f"Trade closed: {trade_id} - {exit_reason.value}, "
                   f"P&L=${trade.realized_pnl:.2f} ({trade.pnl_percent:+.2f}%)")
        
        return trade
    
    def _place_exit_order(self, trade: Trade, exit_price: float) -> bool:
        """Place exit order via API"""
        
        try:
            # Opposite side for exit
            side = "sell" if trade.is_long else "buy"
            
            response = rest_client.place_order(
                symbol=trade.symbol,
                side=side,
                size=int(trade.entry_size),
                order_type="market_order",
                reduce_only=True
            )
            
            if response.get('success') or response.get('result'):
                order_data = response.get('result', {})
                
                trade.exit_order = Order(
                    order_id=str(order_data.get('id', '')),
                    symbol=trade.symbol,
                    side=side,
                    order_type='market',
                    size=trade.entry_size,
                    status='open'
                )
                
                return True
            else:
                logger.error(f"Exit order failed: {response.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"Exception placing exit order: {e}")
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
    
    def get_statistics(self) -> Dict:
        """Get trading statistics"""
        
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
            'max_drawdown': round(self.risk_manager.current_drawdown, 2)
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
