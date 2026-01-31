"""
Trading Signal Generator
Combines technical indicators to generate trade entry/exit signals.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict
import config
from analysis.indicators import IndicatorValues


class SignalType(Enum):
    """Type of trading signal."""
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    NEUTRAL = "NEUTRAL"


class SignalStrength(Enum):
    """Signal strength levels."""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


@dataclass
class TradeSignal:
    """Trading signal with entry/exit details."""
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: int  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    reasons: List[str] = field(default_factory=list)
    indicators: Optional[IndicatorValues] = None
    
    def is_actionable(self) -> bool:
        """Check if signal meets minimum confidence threshold."""
        return self.confidence >= config.SIGNAL_CONFIG['min_confidence']
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0


class SignalGenerator:
    """
    Generates trading signals based on technical indicator analysis.
    Combines multiple indicators to determine entry/exit points.
    """
    
    def __init__(self):
        self.config = config.SIGNAL_CONFIG
        self.rsi_config = config.INDICATOR_CONFIG['rsi']
        
    def _calculate_confidence(self, scores: List[int]) -> int:
        """Calculate overall confidence from individual scores."""
        if not scores:
            return 0
        return int(sum(scores) / len(scores))
    
    def _get_stop_loss(self, entry: float, atr: float, is_long: bool) -> float:
        """Calculate stop loss based on ATR."""
        multiplier = self.config['atr_multiplier_sl']
        if is_long:
            return entry - (atr * multiplier)
        return entry + (atr * multiplier)
    
    def _get_take_profit(self, entry: float, atr: float, is_long: bool) -> float:
        """Calculate take profit based on ATR."""
        multiplier = self.config['atr_multiplier_tp']
        if is_long:
            return entry + (atr * multiplier)
        return entry - (atr * multiplier)
    
    def analyze_long_signals(self, ind: IndicatorValues) -> tuple:
        """
        Analyze indicators for long entry signals.
        Returns: (score, reasons)
        """
        scores = []
        reasons = []
        
        # RSI oversold condition
        if ind.rsi < self.rsi_config['oversold']:
            scores.append(80)
            reasons.append(f"RSI oversold ({ind.rsi:.1f})")
        elif ind.rsi < 40:
            scores.append(60)
            reasons.append(f"RSI low ({ind.rsi:.1f})")
        elif ind.rsi < 50:
            scores.append(40)
        
        # MACD bullish crossover
        if ind.macd_histogram > 0 and ind.macd_line > ind.macd_signal:
            scores.append(75)
            reasons.append("MACD bullish")
        elif ind.macd_histogram > 0:
            scores.append(50)
        
        # Price below lower Bollinger Band
        if ind.bb_percent < 0.1:
            scores.append(85)
            reasons.append("Price at lower BB")
        elif ind.bb_percent < 0.3:
            scores.append(60)
            reasons.append("Price near lower BB")
        
        # EMA alignment (short > medium > long = uptrend)
        if ind.ema_9 > ind.ema_21:
            scores.append(60)
            reasons.append("EMA 9 > 21 (uptrend)")
        
        # Price above EMA (bounce confirmation)
        if ind.price > ind.ema_21 and ind.price > ind.sma_20:
            scores.append(55)
        
        # ADX trend strength
        if ind.adx > 25 and ind.plus_di > ind.minus_di:
            scores.append(70)
            reasons.append(f"ADX strong uptrend ({ind.adx:.1f})")
        
        # Volume confirmation
        if ind.volume_ratio > 1.5:
            scores.append(65)
            reasons.append("High volume")
        
        return self._calculate_confidence(scores) if scores else 0, reasons
    
    def analyze_short_signals(self, ind: IndicatorValues) -> tuple:
        """
        Analyze indicators for short entry signals.
        Returns: (score, reasons)
        """
        scores = []
        reasons = []
        
        # RSI overbought condition
        if ind.rsi > self.rsi_config['overbought']:
            scores.append(80)
            reasons.append(f"RSI overbought ({ind.rsi:.1f})")
        elif ind.rsi > 60:
            scores.append(60)
            reasons.append(f"RSI high ({ind.rsi:.1f})")
        elif ind.rsi > 50:
            scores.append(40)
        
        # MACD bearish crossover
        if ind.macd_histogram < 0 and ind.macd_line < ind.macd_signal:
            scores.append(75)
            reasons.append("MACD bearish")
        elif ind.macd_histogram < 0:
            scores.append(50)
        
        # Price above upper Bollinger Band
        if ind.bb_percent > 0.9:
            scores.append(85)
            reasons.append("Price at upper BB")
        elif ind.bb_percent > 0.7:
            scores.append(60)
            reasons.append("Price near upper BB")
        
        # EMA alignment (short < medium = downtrend)
        if ind.ema_9 < ind.ema_21:
            scores.append(60)
            reasons.append("EMA 9 < 21 (downtrend)")
        
        # Price below EMAs
        if ind.price < ind.ema_21 and ind.price < ind.sma_20:
            scores.append(55)
        
        # ADX trend strength
        if ind.adx > 25 and ind.minus_di > ind.plus_di:
            scores.append(70)
            reasons.append(f"ADX strong downtrend ({ind.adx:.1f})")
        
        # Volume confirmation
        if ind.volume_ratio > 1.5:
            scores.append(65)
            reasons.append("High volume")
        
        return self._calculate_confidence(scores) if scores else 0, reasons
    
    def generate_signal(self, symbol: str, 
                        indicators: IndicatorValues) -> TradeSignal:
        """
        Generate trading signal based on technical indicators.
        
        Args:
            symbol: Trading symbol
            indicators: Calculated indicator values
            
        Returns:
            TradeSignal with recommendations
        """
        # Analyze both directions
        long_score, long_reasons = self.analyze_long_signals(indicators)
        short_score, short_reasons = self.analyze_short_signals(indicators)
        
        entry_price = indicators.price
        atr = indicators.atr if indicators.atr > 0 else entry_price * 0.01
        
        # Determine signal type based on scores
        if long_score >= short_score and long_score >= self.config['min_confidence']:
            signal_type = SignalType.LONG
            confidence = long_score
            reasons = long_reasons
            is_long = True
        elif short_score > long_score and short_score >= self.config['min_confidence']:
            signal_type = SignalType.SHORT
            confidence = short_score
            reasons = short_reasons
            is_long = False
        else:
            # No clear signal
            return TradeSignal(
                symbol=symbol,
                signal_type=SignalType.NEUTRAL,
                strength=SignalStrength.WEAK,
                confidence=max(long_score, short_score),
                entry_price=entry_price,
                stop_loss=0,
                take_profit=0,
                timestamp=datetime.now(),
                reasons=["No clear signal"],
                indicators=indicators
            )
        
        # Calculate SL and TP
        stop_loss = self._get_stop_loss(entry_price, atr, is_long)
        take_profit = self._get_take_profit(entry_price, atr, is_long)
        
        # Determine strength
        if confidence >= 80:
            strength = SignalStrength.STRONG
        elif confidence >= 65:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        return TradeSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=datetime.now(),
            reasons=reasons,
            indicators=indicators
        )
    
    def check_exit_signal(self, symbol: str, 
                          indicators: IndicatorValues,
                          is_long_position: bool,
                          entry_price: float) -> Optional[TradeSignal]:
        """
        Check if current position should be exited.
        
        Args:
            symbol: Trading symbol
            indicators: Current indicator values
            is_long_position: True if holding long, False if short
            entry_price: Original entry price
            
        Returns:
            Exit signal if position should be closed, None otherwise
        """
        reasons = []
        should_exit = False
        
        current_price = indicators.price
        
        if is_long_position:
            # Check long exit conditions
            
            # RSI overbought
            if indicators.rsi > self.rsi_config['overbought']:
                reasons.append(f"RSI overbought ({indicators.rsi:.1f})")
                should_exit = True
            
            # MACD bearish crossover
            if indicators.macd_histogram < 0 and indicators.macd_line < indicators.macd_signal:
                reasons.append("MACD bearish crossover")
                should_exit = True
            
            # Price at upper Bollinger Band
            if indicators.bb_percent > 0.95:
                reasons.append("Price at upper BB")
                should_exit = True
            
            # EMA bearish cross
            if indicators.ema_9 < indicators.ema_21:
                reasons.append("EMA bearish cross")
                should_exit = True
                
        else:  # Short position
            # Check short exit conditions
            
            # RSI oversold
            if indicators.rsi < self.rsi_config['oversold']:
                reasons.append(f"RSI oversold ({indicators.rsi:.1f})")
                should_exit = True
            
            # MACD bullish crossover
            if indicators.macd_histogram > 0 and indicators.macd_line > indicators.macd_signal:
                reasons.append("MACD bullish crossover")
                should_exit = True
            
            # Price at lower Bollinger Band
            if indicators.bb_percent < 0.05:
                reasons.append("Price at lower BB")
                should_exit = True
            
            # EMA bullish cross
            if indicators.ema_9 > indicators.ema_21:
                reasons.append("EMA bullish cross")
                should_exit = True
        
        if should_exit:
            signal_type = SignalType.EXIT_LONG if is_long_position else SignalType.EXIT_SHORT
            pnl_percent = ((current_price - entry_price) / entry_price * 100)
            if not is_long_position:
                pnl_percent = -pnl_percent
            
            reasons.append(f"P&L: {pnl_percent:+.2f}%")
            
            return TradeSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=SignalStrength.MODERATE,
                confidence=70,
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                timestamp=datetime.now(),
                reasons=reasons,
                indicators=indicators
            )
        
        return None


# Singleton instance
signal_generator = SignalGenerator()
