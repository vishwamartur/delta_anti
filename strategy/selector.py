"""
Adaptive Strategy Selector
Dynamically selects the best trading strategy based on market regime.
"""
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from analysis.market_analyzer import MarketRegime, MarketCondition
from analysis.indicators import IndicatorValues
from analysis.signals import TradeSignal, SignalType, SignalStrength

# Import strategies
from analysis.signals import signal_generator  # Default momentum strategy
try:
    from strategy.range_strategy import get_range_strategy
    RANGE_STRATEGY_AVAILABLE = True
except ImportError:
    RANGE_STRATEGY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class StrategyResult:
    """Standardized result from any strategy."""
    signal: TradeSignal
    strategy_name: str
    regime: MarketRegime
    confidence: int

class AdaptiveStrategySelector:
    """
    Selects and executes the appropriate strategy based on market conditions.
    
    Strategies:
    1. TRENDING (UP/DOWN) -> Momentum Strategy (Default)
       - Uses EMAs, MACD, RSI, ADX
       - Best for strong trends (ADX > 25)
       
    2. RANGING -> Mean Reversion Strategy
       - Uses Bollinger Bands, RSI reversals
       - Best for sideways markets (ADX < 20, BB Width < 3%)
       
    3. HIGH VOLATILITY -> Breakout or Cash
       - Uses volatility contraction/expansion
       - Reduces position size or waits for clarity
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.range_strategy = get_range_strategy() if RANGE_STRATEGY_AVAILABLE else None
        
    def select_and_generate(
        self,
        symbol: str,
        indicators: IndicatorValues,
        market_condition: MarketCondition,
        df=None,
        df_htf=None
    ) -> StrategyResult:
        """
        Select best strategy and generate signal.
        """
        regime = market_condition.regime
        
        # Default to neutral signal
        signal = TradeSignal(
            symbol=symbol,
            signal_type=SignalType.NEUTRAL,
            strength=SignalStrength.WEAK,
            confidence=0,
            entry_price=indicators.price,
            stop_loss=0,
            take_profit=0,
            timestamp=market_condition.timestamp
        )
        strategy_name = "Neutral"
        
        try:
            # === STRATEGY SELECTION LOGIC ===
            
            # 1. RANGING REGIME -> Range Strategy
            if regime == MarketRegime.RANGING:
                if self.range_strategy:
                    logger.info(f"[STRATEGY] {symbol}: Ranging market detected. Switching to Mean Reversion.")
                    range_signal = self.range_strategy.generate_signal(symbol, indicators, indicators.price)
                    if range_signal:
                        if range_signal.direction in ('BUY', 'LONG'):
                            signal_type = SignalType.LONG
                        elif range_signal.direction in ('SELL', 'SHORT'):
                            signal_type = SignalType.SHORT
                        else:
                            signal_type = SignalType.NEUTRAL
                            
                        signal = TradeSignal(
                            symbol=symbol,
                            signal_type=signal_type,
                            strength=SignalStrength.MODERATE, 
                            confidence=int(range_signal.confidence),
                            entry_price=indicators.price,
                            stop_loss=range_signal.stop_loss,
                            take_profit=range_signal.take_profit,
                            timestamp=market_condition.timestamp,
                            reasons=[f"Range: {range_signal.reason}"],
                            indicators=indicators
                        )
                        strategy_name = "MeanReversion"
                    else:
                        logger.info(f"[STRATEGY] {symbol}: Range strategy did not generate a signal. Falling back to Momentum.")
                        signal = signal_generator.generate_signal(symbol, indicators, df, df_htf)
                        strategy_name = "Momentum (Fallback)"
                else:
                    logger.warning(f"[STRATEGY] {symbol}: Ranging detected but Range Strategy not available. Using Momentum.")
                    # Fallback to Momentum
                    signal = signal_generator.generate_signal(symbol, indicators, df, df_htf)
                    strategy_name = "Momentum (Fallback)"

            # 2. HIGH VOLATILITY -> Conservative / Cash
            elif regime == MarketRegime.HIGH_VOLATILITY:
                logger.info(f"[STRATEGY] {symbol}: High Volatility detected. Reducing risk.")
                # We can still use momentum but maybe stricter filtering?
                # For now, let's just use momentum but tag it
                signal = signal_generator.generate_signal(symbol, indicators, df, df_htf)
                
                # Penalize confidence due to volatility risk
                signal.confidence = max(0, signal.confidence - 20)
                signal.reasons.append("High Volatility Penalty (-20%)")
                strategy_name = "Momentum (Volatile)"
                
            # 3. TRENDING (or UNCERTAIN) -> Momentum Strategy
            else:
                # Default behavior
                signal = signal_generator.generate_signal(symbol, indicators, df, df_htf)
                strategy_name = "Momentum"
                
        except Exception as e:
            logger.error(f"[STRATEGY] Error selecting strategy for {symbol}: {e}")
            
        return StrategyResult(
            signal=signal,
            strategy_name=strategy_name,
            regime=regime,
            confidence=signal.confidence
        )

# Singleton instance
_selector = None

def get_strategy_selector(config: dict = None):
    global _selector
    if _selector is None:
        _selector = AdaptiveStrategySelector(config)
    return _selector
