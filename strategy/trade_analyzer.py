"""
Trade History Analyzer
Analyzes past trades to learn optimal parameters and improve profitability.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SymbolStats:
    """Statistics for a specific symbol+direction combination"""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_pnl_percent: float = 0.0
    avg_duration_minutes: float = 0.0
    avg_confidence: float = 0.0
    stop_loss_exits: int = 0
    take_profit_exits: int = 0
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate as percentage"""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def is_profitable(self) -> bool:
        """Check if this combo is historically profitable"""
        return self.total_pnl > 0 and self.win_rate >= 30


@dataclass
class AdaptiveParameters:
    """Dynamically adjusted trading parameters"""
    min_confidence: int = 70
    sl_atr_multiplier: float = 1.5
    tp_atr_multiplier: float = 2.0
    blocked_combos: List[Tuple[str, str]] = field(default_factory=list)  # (symbol, direction)
    preferred_combos: List[Tuple[str, str]] = field(default_factory=list)


class TradeHistoryAnalyzer:
    """
    Analyzes trade history to learn optimal parameters.
    
    Features:
    - Track win rate per symbol+direction combo
    - Block trades that historically lose
    - Adjust SL/TP multipliers based on what worked
    - Dynamically raise confidence thresholds
    """
    
    def __init__(self, trades_file: str = "data/trades.json"):
        self.trades_file = Path(trades_file)
        self.symbol_stats: Dict[str, SymbolStats] = {}  # key: "SYMBOL_DIRECTION"
        self.adaptive_params = AdaptiveParameters()
        
        # Config
        self.min_historical_trades = 5  # Need at least N trades to make decisions
        self.min_win_rate_threshold = 30  # Block combos with <30% win rate
        self.min_confidence_default = 70  # Raised from 60
        
        # Load and analyze on init
        self._load_and_analyze()
    
    def _load_and_analyze(self):
        """Load trade history and run analysis"""
        trades = self._load_trades()
        if trades:
            self.analyze_all(trades)
            self._calculate_adaptive_parameters()
            logger.info(f"[ADAPTIVE] Loaded {len(trades)} historical trades")
    
    def _load_trades(self) -> List[Dict]:
        """Load trades from JSON file"""
        if not self.trades_file.exists():
            logger.warning(f"[ADAPTIVE] Trades file not found: {self.trades_file}")
            return []
        
        try:
            with open(self.trades_file, 'r') as f:
                data = json.load(f)
            
            # Get closed trades from history
            history = data.get('history', [])
            trades = data.get('trades', {})
            
            # Combine history with closed trades from main dict
            all_trades = list(history)
            for trade in trades.values():
                if trade.get('state') == 'CLOSED' and trade not in all_trades:
                    all_trades.append(trade)
            
            return all_trades
            
        except Exception as e:
            logger.error(f"[ADAPTIVE] Error loading trades: {e}")
            return []
    
    def analyze_all(self, trades: List[Dict]):
        """Analyze all trades and build statistics"""
        self.symbol_stats.clear()
        
        for trade in trades:
            symbol = trade.get('symbol', '')
            direction = trade.get('direction', '')
            
            if not symbol or not direction:
                continue
            
            key = f"{symbol}_{direction}"
            
            # Initialize stats if needed
            if key not in self.symbol_stats:
                self.symbol_stats[key] = SymbolStats(symbol=symbol, direction=direction)
            
            stats = self.symbol_stats[key]
            stats.total_trades += 1
            
            # Determine win/loss
            pnl = trade.get('realized_pnl', 0) or trade.get('unrealized_pnl', 0)
            pnl_percent = trade.get('pnl_percent', 0)
            
            if pnl > 0:
                stats.winning_trades += 1
            else:
                stats.losing_trades += 1
            
            stats.total_pnl += pnl
            stats.avg_pnl_percent = (
                (stats.avg_pnl_percent * (stats.total_trades - 1) + pnl_percent) 
                / stats.total_trades
            )
            
            # Duration
            duration = trade.get('duration_minutes', 0)
            stats.avg_duration_minutes = (
                (stats.avg_duration_minutes * (stats.total_trades - 1) + duration)
                / stats.total_trades
            )
            
            # Confidence
            confidence = trade.get('ml_confidence', 0)
            stats.avg_confidence = (
                (stats.avg_confidence * (stats.total_trades - 1) + confidence)
                / stats.total_trades
            )
            
            # Exit reasons
            exit_reason = trade.get('exit_reason', '').lower()
            if 'stop' in exit_reason:
                stats.stop_loss_exits += 1
            elif 'profit' in exit_reason or 'take' in exit_reason:
                stats.take_profit_exits += 1
    
    def analyze_performance_by_symbol(self) -> Dict[str, Dict]:
        """Get performance breakdown by symbol"""
        result = defaultdict(lambda: {'total': 0, 'wins': 0, 'pnl': 0.0})
        
        for key, stats in self.symbol_stats.items():
            symbol = stats.symbol
            result[symbol]['total'] += stats.total_trades
            result[symbol]['wins'] += stats.winning_trades
            result[symbol]['pnl'] += stats.total_pnl
        
        # Calculate win rates
        for symbol in result:
            total = result[symbol]['total']
            if total > 0:
                result[symbol]['win_rate'] = (result[symbol]['wins'] / total) * 100
            else:
                result[symbol]['win_rate'] = 0
        
        return dict(result)
    
    def analyze_performance_by_direction(self) -> Dict[str, Dict]:
        """Get performance breakdown by direction"""
        result = defaultdict(lambda: {'total': 0, 'wins': 0, 'pnl': 0.0})
        
        for key, stats in self.symbol_stats.items():
            direction = stats.direction
            result[direction]['total'] += stats.total_trades
            result[direction]['wins'] += stats.winning_trades
            result[direction]['pnl'] += stats.total_pnl
        
        # Calculate win rates
        for direction in result:
            total = result[direction]['total']
            if total > 0:
                result[direction]['win_rate'] = (result[direction]['wins'] / total) * 100
            else:
                result[direction]['win_rate'] = 0
        
        return dict(result)
    
    def calculate_optimal_atr_multipliers(self) -> Dict[str, float]:
        """Calculate optimal ATR multipliers based on trade outcomes"""
        # Analyze stop loss and take profit hit rates
        total_sl_exits = sum(s.stop_loss_exits for s in self.symbol_stats.values())
        total_tp_exits = sum(s.take_profit_exits for s in self.symbol_stats.values())
        total_trades = sum(s.total_trades for s in self.symbol_stats.values())
        
        if total_trades == 0:
            return {'sl_multiplier': 1.5, 'tp_multiplier': 2.0}
        
        sl_hit_rate = (total_sl_exits / total_trades) * 100
        tp_hit_rate = (total_tp_exits / total_trades) * 100
        
        # If stopped out too often (>50%), widen stops
        sl_multiplier = 1.5
        if sl_hit_rate > 50:
            sl_multiplier = 2.0  # Widen stop loss
            logger.info(f"[ADAPTIVE] High SL hit rate ({sl_hit_rate:.1f}%), widening to 2.0x ATR")
        elif sl_hit_rate < 20:
            sl_multiplier = 1.2  # Tighten stop loss
        
        # If TP rarely hit, tighten it
        tp_multiplier = 2.0
        if tp_hit_rate < 30:
            tp_multiplier = 1.5  # Tighten take profit
            logger.info(f"[ADAPTIVE] Low TP hit rate ({tp_hit_rate:.1f}%), tightening to 1.5x ATR")
        elif tp_hit_rate > 60:
            tp_multiplier = 2.5  # Let winners run
        
        return {
            'sl_multiplier': sl_multiplier,
            'tp_multiplier': tp_multiplier,
            'sl_hit_rate': sl_hit_rate,
            'tp_hit_rate': tp_hit_rate
        }
    
    def _calculate_adaptive_parameters(self):
        """Calculate and update adaptive parameters"""
        blocked = []
        preferred = []
        
        for key, stats in self.symbol_stats.items():
            # Only make decisions with enough data
            if stats.total_trades < self.min_historical_trades:
                continue
            
            # Block losing combos
            if stats.win_rate < self.min_win_rate_threshold:
                blocked.append((stats.symbol, stats.direction))
                logger.info(f"[ADAPTIVE] Blocking {stats.symbol} {stats.direction} "
                           f"(win rate: {stats.win_rate:.1f}%)")
            
            # Mark preferred combos
            elif stats.win_rate >= 50 and stats.total_pnl > 0:
                preferred.append((stats.symbol, stats.direction))
                logger.info(f"[ADAPTIVE] Preferring {stats.symbol} {stats.direction} "
                           f"(win rate: {stats.win_rate:.1f}%, PnL: ${stats.total_pnl:.2f})")
        
        # Get optimal multipliers
        multipliers = self.calculate_optimal_atr_multipliers()
        
        # Calculate optimal confidence threshold
        # Analyze winning vs losing trade confidence
        winning_confidences = []
        losing_confidences = []
        
        trades = self._load_trades()
        for trade in trades:
            conf = trade.get('ml_confidence', 0)
            pnl = trade.get('realized_pnl', 0) or trade.get('unrealized_pnl', 0)
            if pnl > 0:
                winning_confidences.append(conf)
            else:
                losing_confidences.append(conf)
        
        # Set min confidence to avg of winning trades (if we have data)
        min_confidence = self.min_confidence_default
        if winning_confidences:
            avg_winning_conf = sum(winning_confidences) / len(winning_confidences)
            min_confidence = max(self.min_confidence_default, int(avg_winning_conf - 5))
            logger.info(f"[ADAPTIVE] Min confidence set to {min_confidence} "
                       f"(avg winning: {avg_winning_conf:.0f})")
        
        # Update adaptive params
        self.adaptive_params = AdaptiveParameters(
            min_confidence=min_confidence,
            sl_atr_multiplier=multipliers['sl_multiplier'],
            tp_atr_multiplier=multipliers['tp_multiplier'],
            blocked_combos=blocked,
            preferred_combos=preferred
        )
    
    def get_adaptive_parameters(self) -> Dict:
        """Get current adaptive parameters as dict"""
        return {
            'min_confidence': self.adaptive_params.min_confidence,
            'sl_atr_multiplier': self.adaptive_params.sl_atr_multiplier,
            'tp_atr_multiplier': self.adaptive_params.tp_atr_multiplier,
            'blocked_combos': self.adaptive_params.blocked_combos,
            'preferred_combos': self.adaptive_params.preferred_combos
        }
    
    def should_take_trade(self, symbol: str, direction: str, confidence: int) -> Tuple[bool, str]:
        """
        Check if a trade should be taken based on historical performance.
        
        Returns:
            Tuple of (should_take, reason)
        """
        # Check confidence threshold
        if confidence < self.adaptive_params.min_confidence:
            return False, f"Confidence {confidence} < min {self.adaptive_params.min_confidence}"
        
        # Check if combo is blocked
        combo = (symbol, direction)
        if combo in self.adaptive_params.blocked_combos:
            return False, f"{symbol} {direction} blocked due to poor historical performance"
        
        # All checks passed
        reason = "Trade approved"
        if combo in self.adaptive_params.preferred_combos:
            reason = f"Trade approved (preferred combo: high historical performance)"
        
        return True, reason
    
    def get_performance_report(self) -> str:
        """Generate a human-readable performance report"""
        lines = [
            "=" * 60,
            "TRADE HISTORY ANALYSIS REPORT",
            "=" * 60,
            ""
        ]
        
        # Overall stats
        total_trades = sum(s.total_trades for s in self.symbol_stats.values())
        total_wins = sum(s.winning_trades for s in self.symbol_stats.values())
        total_pnl = sum(s.total_pnl for s in self.symbol_stats.values())
        
        if total_trades > 0:
            overall_win_rate = (total_wins / total_trades) * 100
        else:
            overall_win_rate = 0
        
        lines.extend([
            f"Overall Performance:",
            f"  Total Trades: {total_trades}",
            f"  Win Rate: {overall_win_rate:.1f}%",
            f"  Total PnL: ${total_pnl:.2f}",
            ""
        ])
        
        # By symbol+direction
        lines.append("Performance by Symbol+Direction:")
        for key, stats in sorted(self.symbol_stats.items()):
            status = "[WIN]" if stats.is_profitable else "[LOSS]"
            lines.append(
                f"  {status} {stats.symbol} {stats.direction}: "
                f"{stats.total_trades} trades, {stats.win_rate:.0f}% win, "
                f"${stats.total_pnl:.2f} PnL"
            )
        
        lines.extend([
            "",
            "Adaptive Parameters:",
            f"  Min Confidence: {self.adaptive_params.min_confidence}",
            f"  SL ATR Multiplier: {self.adaptive_params.sl_atr_multiplier}x",
            f"  TP ATR Multiplier: {self.adaptive_params.tp_atr_multiplier}x",
        ])
        
        if self.adaptive_params.blocked_combos:
            lines.append(f"  BLOCKED: {self.adaptive_params.blocked_combos}")
        
        if self.adaptive_params.preferred_combos:
            lines.append(f"  PREFERRED: {self.adaptive_params.preferred_combos}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def refresh(self):
        """Reload trade history and recalculate parameters"""
        self._load_and_analyze()


# Singleton instance
analyzer = TradeHistoryAnalyzer()
