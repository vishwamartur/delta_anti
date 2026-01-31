"""
Trade Manager - Tracks active trades and monitors for exit conditions
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import config
from analysis.signals import TradeSignal, SignalType


class TradeStatus(Enum):
    """Status of a trade."""
    PENDING = "PENDING"      # Signal generated, not yet executed
    OPEN = "OPEN"            # Trade is active
    CLOSED = "CLOSED"        # Trade has been closed
    CANCELLED = "CANCELLED"  # Trade was cancelled


@dataclass
class Trade:
    """Represents a single trade."""
    id: str
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    size: int
    stop_loss: float
    take_profit: float
    status: TradeStatus
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    exit_reason: str = ""
    signal: Optional[TradeSignal] = None
    
    @property
    def is_long(self) -> bool:
        return self.side == "long"
    
    @property
    def duration_minutes(self) -> int:
        """Trade duration in minutes."""
        end = self.exit_time or datetime.now()
        return int((end - self.entry_time).total_seconds() / 60)
    
    def update_pnl(self, current_price: float):
        """Update unrealized P&L."""
        if self.is_long:
            self.pnl = (current_price - self.entry_price) * self.size
            self.pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            self.pnl = (self.entry_price - current_price) * self.size
            self.pnl_percent = ((self.entry_price - current_price) / self.entry_price) * 100
    
    def check_sl_tp(self, current_price: float) -> tuple:
        """
        Check if stop loss or take profit is hit.
        Returns: (is_hit, reason)
        """
        if self.is_long:
            if current_price <= self.stop_loss:
                return True, "Stop Loss Hit"
            if current_price >= self.take_profit:
                return True, "Take Profit Hit"
        else:
            if current_price >= self.stop_loss:
                return True, "Stop Loss Hit"
            if current_price <= self.take_profit:
                return True, "Take Profit Hit"
        return False, ""


class TradeManager:
    """
    Manages all trades - tracking, monitoring, and history.
    """
    
    def __init__(self):
        self._trades: Dict[str, Trade] = {}  # id -> Trade
        self._open_trades: Dict[str, Trade] = {}  # symbol -> Trade (one per symbol)
        self._trade_history: List[Trade] = []
        self._trade_counter = 0
        
        # Daily stats
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._last_reset_date = datetime.now().date()
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return f"T{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._trade_counter}"
    
    def _reset_daily_stats(self):
        """Reset daily statistics if new day."""
        today = datetime.now().date()
        if today != self._last_reset_date:
            self._daily_trades = 0
            self._daily_pnl = 0.0
            self._last_reset_date = today
    
    def has_open_position(self, symbol: str) -> bool:
        """Check if there's an open position for symbol."""
        return symbol in self._open_trades
    
    def get_open_trade(self, symbol: str) -> Optional[Trade]:
        """Get open trade for symbol."""
        return self._open_trades.get(symbol)
    
    def create_trade_from_signal(self, signal: TradeSignal, 
                                  size: int = None) -> Optional[Trade]:
        """
        Create a new trade from a signal.
        
        Args:
            signal: Trade signal with entry/SL/TP
            size: Position size (uses config default if not provided)
            
        Returns:
            Created Trade object or None if trade not allowed
        """
        self._reset_daily_stats()
        
        # Check if we can take this trade
        if self.has_open_position(signal.symbol):
            return None
        
        if self._daily_trades >= config.MAX_DAILY_TRADES:
            return None
        
        if signal.signal_type == SignalType.NEUTRAL:
            return None
        
        size = size or config.MAX_POSITION_SIZE
        side = "long" if signal.signal_type == SignalType.LONG else "short"
        
        trade = Trade(
            id=self._generate_trade_id(),
            symbol=signal.symbol,
            side=side,
            entry_price=signal.entry_price,
            size=size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            status=TradeStatus.PENDING,
            entry_time=datetime.now(),
            signal=signal
        )
        
        self._trades[trade.id] = trade
        return trade
    
    def open_trade(self, trade_id: str, actual_entry_price: float = None):
        """Mark trade as open (executed)."""
        if trade_id not in self._trades:
            return
        
        trade = self._trades[trade_id]
        
        if actual_entry_price:
            trade.entry_price = actual_entry_price
        
        trade.status = TradeStatus.OPEN
        trade.entry_time = datetime.now()
        self._open_trades[trade.symbol] = trade
        self._daily_trades += 1
        
        print(f"[TRADE] Opened {trade.side.upper()} {trade.symbol} @ {trade.entry_price}")
    
    def close_trade(self, symbol: str, exit_price: float, reason: str = ""):
        """Close an open trade."""
        if symbol not in self._open_trades:
            return None
        
        trade = self._open_trades[symbol]
        trade.status = TradeStatus.CLOSED
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        trade.exit_reason = reason
        
        # Calculate final P&L
        trade.update_pnl(exit_price)
        self._daily_pnl += trade.pnl
        
        # Move to history
        self._trade_history.append(trade)
        del self._open_trades[symbol]
        
        print(f"[TRADE] Closed {trade.side.upper()} {symbol} @ {exit_price} "
              f"| P&L: {trade.pnl_percent:+.2f}% | Reason: {reason}")
        
        return trade
    
    def update_all_trades(self, prices: Dict[str, float]):
        """Update P&L for all open trades."""
        for symbol, trade in self._open_trades.items():
            if symbol in prices:
                trade.update_pnl(prices[symbol])
    
    def check_exit_conditions(self, symbol: str, 
                               current_price: float) -> Optional[str]:
        """
        Check if trade should be exited based on SL/TP.
        
        Returns:
            Exit reason if should exit, None otherwise
        """
        trade = self._open_trades.get(symbol)
        if not trade:
            return None
        
        is_hit, reason = trade.check_sl_tp(current_price)
        return reason if is_hit else None
    
    @property
    def open_trade_count(self) -> int:
        return len(self._open_trades)
    
    @property
    def open_trades(self) -> List[Trade]:
        return list(self._open_trades.values())
    
    @property
    def daily_trades(self) -> int:
        self._reset_daily_stats()
        return self._daily_trades
    
    @property
    def daily_pnl(self) -> float:
        self._reset_daily_stats()
        return self._daily_pnl
    
    @property
    def total_unrealized_pnl(self) -> float:
        return sum(t.pnl for t in self._open_trades.values())
    
    def get_trade_history(self, limit: int = 20) -> List[Trade]:
        """Get recent trade history."""
        return self._trade_history[-limit:]
    
    def get_stats(self) -> Dict:
        """Get trading statistics."""
        self._reset_daily_stats()
        
        if not self._trade_history:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "total_pnl": 0,
                "daily_trades": self._daily_trades,
                "daily_pnl": self._daily_pnl
            }
        
        closed = [t for t in self._trade_history if t.status == TradeStatus.CLOSED]
        winners = [t for t in closed if t.pnl > 0]
        
        return {
            "total_trades": len(closed),
            "win_rate": (len(winners) / len(closed) * 100) if closed else 0,
            "avg_pnl": sum(t.pnl_percent for t in closed) / len(closed) if closed else 0,
            "total_pnl": sum(t.pnl for t in closed),
            "daily_trades": self._daily_trades,
            "daily_pnl": self._daily_pnl
        }


# Singleton instance
trade_manager = TradeManager()
