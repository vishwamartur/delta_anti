"""
Low Volatility Range Trading Strategy
Profits from price oscillations in slow/sideways markets.

Features:
- Auto-detects low volatility (ranging) markets
- Mean reversion entries near Bollinger Band edges
- Scalping mode with tight take profits
- Works when momentum strategies fail
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"          # Low volatility, sideways
    HIGH_VOLATILITY = "high_vol" # Choppy, avoid trading


@dataclass
class RangeTradeSignal:
    """Signal for range/scalping trades."""
    symbol: str
    direction: str              # 'LONG' or 'SHORT'
    entry_price: float
    take_profit: float
    stop_loss: float
    confidence: float           # 0-100
    regime: MarketRegime
    reason: str
    timestamp: datetime


class LowVolatilityStrategy:
    """
    Strategy for profiting in slow, ranging markets.
    
    Approach:
    1. Detect when market is ranging (low ADX, price in BB range)
    2. Buy at lower BB, sell at upper BB (mean reversion)
    3. Use tight take profits for quick scalping gains
    4. Exit quickly if trend breaks out
    """
    
    def __init__(
        self,
        # Range detection
        adx_threshold: float = 25.0,       # Below this = ranging
        bb_squeeze_pct: float = 2.0,       # BB width % for squeeze
        
        # Entry parameters
        bb_entry_pct: float = 0.8,         # Enter at 80% toward band
        rsi_oversold: float = 35,          # RSI level for longs
        rsi_overbought: float = 65,        # RSI level for shorts
        
        # Take profit (scalping)
        scalp_tp_pct: float = 0.3,         # 0.3% take profit in range
        range_tp_pct: float = 0.6,         # 0.6% TP for mean reversion
        
        # Stop loss
        range_sl_pct: float = 0.4,         # 0.4% stop loss
        
        # Risk management
        max_trades_in_range: int = 5,      # Max concurrent trades
        min_range_duration: int = 10,      # Min candles in range
    ):
        self.adx_threshold = adx_threshold
        self.bb_squeeze_pct = bb_squeeze_pct
        self.bb_entry_pct = bb_entry_pct
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.scalp_tp_pct = scalp_tp_pct
        self.range_tp_pct = range_tp_pct
        self.range_sl_pct = range_sl_pct
        self.max_trades_in_range = max_trades_in_range
        self.min_range_duration = min_range_duration
        
        # State tracking
        self.range_candle_count = 0
        self.last_regime = MarketRegime.RANGING
        self.active_range_trades = 0
        
        logger.info(f"[RANGE] Strategy initialized: ADX<{adx_threshold}, "
                   f"TP={scalp_tp_pct}%/{range_tp_pct}%, SL={range_sl_pct}%")
    
    def detect_regime(self, indicators) -> MarketRegime:
        """
        Detect current market regime.
        
        Args:
            indicators: IndicatorValues from analysis
            
        Returns:
            MarketRegime classification
        """
        adx = getattr(indicators, 'adx', 20)
        bb_width = getattr(indicators, 'bb_width_pct', 3.0)
        rsi = getattr(indicators, 'rsi', 50)
        macd = getattr(indicators, 'macd', 0)
        macd_signal = getattr(indicators, 'macd_signal', 0)
        macd_hist = macd - macd_signal if macd_signal else 0
        
        # Low ADX = no strong trend
        is_low_adx = adx < self.adx_threshold
        
        # Narrow BB = low volatility squeeze
        is_squeeze = bb_width < self.bb_squeeze_pct
        
        # MACD near zero = no momentum
        is_no_momentum = abs(macd_hist) < 0.1 * abs(macd) if macd else True
        
        if is_low_adx and (is_squeeze or is_no_momentum):
            self.range_candle_count += 1
            
            if self.range_candle_count >= self.min_range_duration:
                regime = MarketRegime.RANGING
                logger.debug(f"[RANGE] Detected: ADX={adx:.1f}, BB_width={bb_width:.2f}%")
            else:
                regime = self.last_regime
        elif adx > 40:
            self.range_candle_count = 0
            if macd_hist > 0:
                regime = MarketRegime.TRENDING_UP
            else:
                regime = MarketRegime.TRENDING_DOWN
        else:
            self.range_candle_count = max(0, self.range_candle_count - 1)
            regime = MarketRegime.HIGH_VOLATILITY
        
        if regime != self.last_regime:
            logger.info(f"[REGIME] Changed: {self.last_regime.value} → {regime.value}")
        
        self.last_regime = regime
        return regime
    
    def get_bb_position(self, price: float, bb_upper: float, bb_lower: float) -> float:
        """
        Calculate price position within Bollinger Bands.
        Returns: 0.0 (at lower) to 1.0 (at upper)
        """
        if bb_upper == bb_lower:
            return 0.5
        return (price - bb_lower) / (bb_upper - bb_lower)
    
    def generate_signal(
        self, 
        symbol: str, 
        indicators, 
        current_price: float
    ) -> Optional[RangeTradeSignal]:
        """
        Generate trading signal for range/scalping.
        
        Args:
            symbol: Trading symbol
            indicators: Technical indicators
            current_price: Current price
            
        Returns:
            RangeTradeSignal if conditions met, None otherwise
        """
        # Detect market regime
        regime = self.detect_regime(indicators)
        
        # Only trade in ranging markets
        if regime != MarketRegime.RANGING:
            return None
        
        # Check max trades limit
        if self.active_range_trades >= self.max_trades_in_range:
            return None
        
        # Get indicator values
        rsi = getattr(indicators, 'rsi', 50)
        bb_upper = getattr(indicators, 'bb_upper', current_price * 1.02)
        bb_lower = getattr(indicators, 'bb_lower', current_price * 0.98)
        bb_middle = getattr(indicators, 'bb_middle', current_price)
        
        # Calculate BB position (0 = lower, 1 = upper)
        bb_pos = self.get_bb_position(current_price, bb_upper, bb_lower)
        
        signal = None
        
        # === LONG ENTRY: Price near lower BB + RSI oversold ===
        if bb_pos < (1 - self.bb_entry_pct) and rsi < self.rsi_oversold:
            # Mean reversion LONG
            take_profit = bb_middle  # Target middle of range
            tp_pct = ((take_profit - current_price) / current_price) * 100
            
            # Use scalping TP if target is too far
            if tp_pct > self.range_tp_pct:
                take_profit = current_price * (1 + self.range_tp_pct / 100)
            
            stop_loss = current_price * (1 - self.range_sl_pct / 100)
            
            confidence = min(90, 50 + (self.rsi_oversold - rsi) * 2)
            
            signal = RangeTradeSignal(
                symbol=symbol,
                direction="LONG",
                entry_price=current_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                confidence=confidence,
                regime=regime,
                reason=f"BB lower bounce + RSI oversold ({rsi:.0f})",
                timestamp=datetime.now()
            )
            
            logger.info(f"[RANGE] LONG signal: {symbol} @ ${current_price:.2f}, "
                       f"TP=${take_profit:.2f}, SL=${stop_loss:.2f}")
        
        # === SHORT ENTRY: Price near upper BB + RSI overbought ===
        elif bb_pos > self.bb_entry_pct and rsi > self.rsi_overbought:
            # Mean reversion SHORT
            take_profit = bb_middle  # Target middle of range
            tp_pct = ((current_price - take_profit) / current_price) * 100
            
            # Use scalping TP if target is too far
            if tp_pct > self.range_tp_pct:
                take_profit = current_price * (1 - self.range_tp_pct / 100)
            
            stop_loss = current_price * (1 + self.range_sl_pct / 100)
            
            confidence = min(90, 50 + (rsi - self.rsi_overbought) * 2)
            
            signal = RangeTradeSignal(
                symbol=symbol,
                direction="SHORT",
                entry_price=current_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                confidence=confidence,
                regime=regime,
                reason=f"BB upper rejection + RSI overbought ({rsi:.0f})",
                timestamp=datetime.now()
            )
            
            logger.info(f"[RANGE] SHORT signal: {symbol} @ ${current_price:.2f}, "
                       f"TP=${take_profit:.2f}, SL=${stop_loss:.2f}")
        
        # === SCALPING MODE: Quick profits near mid-range ===
        elif 0.3 < bb_pos < 0.7:
            # Only scalp if strong RSI signal
            if rsi < 30:
                # Oversold scalp LONG
                take_profit = current_price * (1 + self.scalp_tp_pct / 100)
                stop_loss = current_price * (1 - self.range_sl_pct / 100)
                
                signal = RangeTradeSignal(
                    symbol=symbol,
                    direction="LONG",
                    entry_price=current_price,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    confidence=60,
                    regime=regime,
                    reason=f"Scalp LONG - extreme RSI ({rsi:.0f})",
                    timestamp=datetime.now()
                )
                
                logger.info(f"[SCALP] LONG: {symbol} @ ${current_price:.2f}, "
                           f"TP +{self.scalp_tp_pct}%")
                
            elif rsi > 70:
                # Overbought scalp SHORT
                take_profit = current_price * (1 - self.scalp_tp_pct / 100)
                stop_loss = current_price * (1 + self.range_sl_pct / 100)
                
                signal = RangeTradeSignal(
                    symbol=symbol,
                    direction="SHORT",
                    entry_price=current_price,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    confidence=60,
                    regime=regime,
                    reason=f"Scalp SHORT - extreme RSI ({rsi:.0f})",
                    timestamp=datetime.now()
                )
                
                logger.info(f"[SCALP] SHORT: {symbol} @ ${current_price:.2f}, "
                           f"TP -{self.scalp_tp_pct}%")
        
        return signal
    
    def check_exit(
        self,
        trade_direction: str,
        entry_price: float,
        current_price: float,
        indicators
    ) -> Tuple[bool, str]:
        """
        Check if range trade should be exited early.
        
        Returns:
            (should_exit, reason)
        """
        # Check for breakout (regime change)
        regime = self.detect_regime(indicators)
        
        if regime != MarketRegime.RANGING:
            # Market broke out of range - exit immediately
            if trade_direction == "LONG" and regime == MarketRegime.TRENDING_DOWN:
                return True, "Range breakout DOWN - exit long"
            if trade_direction == "SHORT" and regime == MarketRegime.TRENDING_UP:
                return True, "Range breakout UP - exit short"
        
        # Check for adverse move
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        if trade_direction == "SHORT":
            pnl_pct = -pnl_pct
        
        # Quick profit taking in ranging market
        if pnl_pct > self.scalp_tp_pct / 2:
            # In ranging market, take partial profits early
            return False, ""  # Don't exit but could partial close
        
        return False, ""
    
    def on_trade_opened(self):
        """Called when a range trade is opened."""
        self.active_range_trades += 1
    
    def on_trade_closed(self):
        """Called when a range trade is closed."""
        self.active_range_trades = max(0, self.active_range_trades - 1)


# Singleton instance
range_strategy = LowVolatilityStrategy()


def get_range_strategy() -> LowVolatilityStrategy:
    """Get the range trading strategy instance."""
    return range_strategy
