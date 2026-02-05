"""
Pre-Trade Market Analysis Module
Comprehensive AI-driven analysis of market conditions before trade entry.

Features:
- Market regime detection (trending/ranging/volatile)
- Multi-indicator trend confidence scoring
- Support/resistance level identification
- Optimal entry/exit zone calculation
- Trade quality assessment
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from enum import Enum

from analysis.indicators import IndicatorValues
import config

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    UNCERTAIN = "uncertain"


@dataclass
class MarketCondition:
    """
    Comprehensive market condition assessment.
    Contains all analysis results for pre-trade evaluation.
    """
    symbol: str
    timestamp: datetime
    
    # Market regime
    regime: MarketRegime = MarketRegime.UNCERTAIN
    regime_confidence: float = 0.0
    
    # Trend analysis
    trend_direction: str = "neutral"  # "bullish", "bearish", "neutral"
    trend_confidence: int = 0  # 0-100
    trend_strength: float = 0.0  # ADX value
    
    # Structure analysis
    structure_quality: int = 0  # 0-100 (how clear is the market structure)
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    
    # Entry/Exit zones
    optimal_entry_zone: Optional[Tuple[float, float]] = None  # (min, max)
    optimal_exit_zone: Optional[Tuple[float, float]] = None   # (min, max)
    
    # Trade assessment
    risk_reward_ratio: float = 0.0
    trade_quality_score: int = 0  # 0-100
    confidence_adjustment: int = 0  # Adjustment to signal confidence
    
    # Decision
    should_enter: bool = False
    block_reason: str = ""
    analysis_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "regime": self.regime.value,
            "trend_direction": self.trend_direction,
            "trend_confidence": self.trend_confidence,
            "structure_quality": self.structure_quality,
            "support_levels": self.support_levels[:3],  # Top 3
            "resistance_levels": self.resistance_levels[:3],
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "trade_quality_score": self.trade_quality_score,
            "should_enter": self.should_enter,
            "block_reason": self.block_reason,
        }


class MarketAnalyzer:
    """
    Comprehensive pre-trade market analysis engine.
    
    Analyzes market conditions before trade entry to:
    1. Detect market regime (trending/ranging/volatile)
    2. Calculate trend confidence with multi-indicator scoring
    3. Identify support/resistance levels
    4. Calculate optimal entry/exit zones
    5. Assess overall trade quality
    """
    
    def __init__(self):
        self.config = getattr(config, 'MARKET_ANALYSIS_CONFIG', {
            "enabled": True,
            "min_trade_quality": 60,
            "min_trend_confidence": 50,
            "sr_lookback_periods": 50,
            "min_rr_ratio": 1.5,
            "use_ai_confirmation": True,
        })
        
        # ML components (lazy loaded)
        self._ml_predictor = None
        self._sentiment_analyzer = None
        
    def analyze_market(
        self, 
        symbol: str, 
        df: pd.DataFrame, 
        indicators: IndicatorValues,
        direction: str = None
    ) -> MarketCondition:
        """
        Perform comprehensive market analysis.
        
        Args:
            symbol: Trading symbol
            df: OHLCV DataFrame
            indicators: Pre-calculated indicator values
            direction: Optional trade direction to validate ("LONG" or "SHORT")
            
        Returns:
            MarketCondition with full analysis results
        """
        condition = MarketCondition(
            symbol=symbol,
            timestamp=datetime.now()
        )
        
        if df is None or len(df) < 20:
            condition.block_reason = "Insufficient data"
            condition.should_enter = False
            return condition
        
        try:
            # 1. Detect market regime
            condition.regime, condition.regime_confidence = self._detect_regime(indicators, df)
            condition.analysis_reasons.append(f"Regime: {condition.regime.value} ({condition.regime_confidence:.0f}%)")
            
            # 2. Calculate trend confidence
            condition.trend_direction, condition.trend_confidence = self._calculate_trend_confidence(indicators, df)
            condition.trend_strength = indicators.adx
            condition.analysis_reasons.append(f"Trend: {condition.trend_direction} ({condition.trend_confidence}%)")
            
            # 3. Find support/resistance levels
            condition.support_levels, condition.resistance_levels = self._find_support_resistance(df)
            
            # 4. Calculate structure quality
            condition.structure_quality = self._calculate_structure_quality(
                indicators, condition.support_levels, condition.resistance_levels
            )
            
            # 5. Calculate optimal entry/exit zones
            current_price = indicators.price
            condition.optimal_entry_zone = self._calculate_entry_zone(
                current_price, condition.support_levels, condition.resistance_levels,
                direction or condition.trend_direction
            )
            condition.optimal_exit_zone = self._calculate_exit_zone(
                current_price, condition.support_levels, condition.resistance_levels,
                direction or condition.trend_direction, indicators.atr
            )
            
            # 6. Calculate risk/reward ratio
            condition.risk_reward_ratio = self._calculate_risk_reward(
                current_price, condition.optimal_entry_zone, 
                condition.optimal_exit_zone, condition.support_levels,
                direction or condition.trend_direction
            )
            condition.analysis_reasons.append(f"R:R = {condition.risk_reward_ratio:.2f}")
            
            # 7. Get AI confirmation if available
            ai_confirmation, ai_confidence = self._get_ai_confirmation(symbol, df, direction)
            if ai_confirmation is not None:
                if ai_confirmation:
                    condition.confidence_adjustment += 10
                    condition.analysis_reasons.append(f"AI confirms ({ai_confidence}%)")
                else:
                    condition.confidence_adjustment -= 15
                    condition.analysis_reasons.append(f"AI conflicts ({ai_confidence}%)")
            
            # 8. Calculate overall trade quality score
            condition.trade_quality_score = self._calculate_trade_quality(condition, direction)
            
            # 9. Make final decision
            condition.should_enter, condition.block_reason = self._make_decision(
                condition, direction
            )
            
            logger.info(f"[MARKET] {symbol}: {condition.regime.value}, "
                       f"quality={condition.trade_quality_score}, "
                       f"enter={condition.should_enter}")
            
        except Exception as e:
            logger.error(f"[MARKET] Analysis error for {symbol}: {e}")
            condition.block_reason = f"Analysis error: {str(e)[:50]}"
            condition.should_enter = False
            
        return condition
    
    def _detect_regime(
        self, 
        indicators: IndicatorValues, 
        df: pd.DataFrame
    ) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime.
        
        Uses:
        - ADX for trend strength
        - BB width for volatility
        - Price action pattern
        """
        adx = indicators.adx
        bb_width = indicators.bb_width if hasattr(indicators, 'bb_width') else 0
        plus_di = indicators.plus_di
        minus_di = indicators.minus_di
        
        # Normalize BB width to percentage
        if bb_width == 0 and len(df) > 20:
            close = df['close'].iloc[-1]
            bb_upper = df['close'].rolling(20).mean().iloc[-1] + 2 * df['close'].rolling(20).std().iloc[-1]
            bb_lower = df['close'].rolling(20).mean().iloc[-1] - 2 * df['close'].rolling(20).std().iloc[-1]
            bb_width = ((bb_upper - bb_lower) / close) * 100 if close > 0 else 2
        
        # High volatility regime
        if bb_width > 5.0:  # BB width > 5% = high volatility
            return MarketRegime.HIGH_VOLATILITY, 80.0
        
        # Trending regime
        if adx > 25:
            confidence = min(95, 60 + (adx - 25) * 1.5)
            if plus_di > minus_di:
                return MarketRegime.TRENDING_UP, confidence
            else:
                return MarketRegime.TRENDING_DOWN, confidence
        
        # Ranging regime
        if adx < 20 and bb_width < 3.0:
            confidence = min(90, 50 + (20 - adx) * 2)
            return MarketRegime.RANGING, confidence
        
        # Uncertain
        return MarketRegime.UNCERTAIN, 40.0
    
    def _calculate_trend_confidence(
        self, 
        indicators: IndicatorValues,
        df: pd.DataFrame
    ) -> Tuple[str, int]:
        """
        Calculate trend direction and confidence using multiple indicators.
        
        Scoring:
        - EMA alignment: +20
        - MACD direction: +20
        - RSI position: +15
        - ADX strength: +20
        - Price action: +25
        """
        bullish_score = 0
        bearish_score = 0
        
        # 1. EMA alignment (20 points)
        if indicators.ema_9 > indicators.ema_21 > indicators.ema_50:
            bullish_score += 20
        elif indicators.ema_9 < indicators.ema_21 < indicators.ema_50:
            bearish_score += 20
        elif indicators.ema_9 > indicators.ema_21:
            bullish_score += 10
        elif indicators.ema_9 < indicators.ema_21:
            bearish_score += 10
        
        # 2. MACD direction (20 points)
        if indicators.macd_histogram > 0 and indicators.macd_line > indicators.macd_signal:
            bullish_score += 20
        elif indicators.macd_histogram < 0 and indicators.macd_line < indicators.macd_signal:
            bearish_score += 20
        elif indicators.macd_histogram > 0:
            bullish_score += 10
        elif indicators.macd_histogram < 0:
            bearish_score += 10
        
        # 3. RSI position (15 points)
        if indicators.rsi > 50:
            bullish_score += int((indicators.rsi - 50) * 0.3)  # Max 15 at RSI 100
        else:
            bearish_score += int((50 - indicators.rsi) * 0.3)
        
        # 4. ADX with DI (20 points)
        if indicators.adx > 20:
            strength_bonus = min(20, (indicators.adx - 20) * 1.0)
            if indicators.plus_di > indicators.minus_di:
                bullish_score += int(strength_bonus)
            else:
                bearish_score += int(strength_bonus)
        
        # 5. Price action - recent momentum (25 points)
        if len(df) >= 10:
            recent_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100
            if recent_change > 0:
                bullish_score += min(25, int(recent_change * 10))
            else:
                bearish_score += min(25, int(abs(recent_change) * 10))
        
        # Determine direction and confidence
        if bullish_score > bearish_score:
            direction = "bullish"
            confidence = min(100, bullish_score)
        elif bearish_score > bullish_score:
            direction = "bearish"
            confidence = min(100, bearish_score)
        else:
            direction = "neutral"
            confidence = 50
        
        return direction, confidence
    
    def _find_support_resistance(
        self, 
        df: pd.DataFrame,
        lookback: int = None
    ) -> Tuple[List[float], List[float]]:
        """
        Find support and resistance levels using swing highs/lows.
        
        Returns:
            Tuple of (support_levels, resistance_levels) sorted by proximity to current price
        """
        lookback = lookback or self.config.get('sr_lookback_periods', 50)
        lookback = min(lookback, len(df) - 1)
        
        if lookback < 10:
            return [], []
        
        recent_df = df.iloc[-lookback:].copy()
        current_price = recent_df['close'].iloc[-1]
        
        supports = []
        resistances = []
        
        # Find swing highs and lows (using 5-period window)
        for i in range(5, len(recent_df) - 5):
            high = recent_df['high'].iloc[i]
            low = recent_df['low'].iloc[i]
            
            # Swing high (resistance)
            if high == recent_df['high'].iloc[i-5:i+6].max():
                resistances.append(high)
            
            # Swing low (support)
            if low == recent_df['low'].iloc[i-5:i+6].min():
                supports.append(low)
        
        # Add recent highs/lows as potential levels
        recent_high = recent_df['high'].max()
        recent_low = recent_df['low'].min()
        
        if recent_high not in resistances:
            resistances.append(recent_high)
        if recent_low not in supports:
            supports.append(recent_low)
        
        # Filter: only levels above/below current price
        supports = [s for s in supports if s < current_price]
        resistances = [r for r in resistances if r > current_price]
        
        # Sort by proximity to current price
        supports = sorted(supports, reverse=True)[:5]  # Closest first
        resistances = sorted(resistances)[:5]  # Closest first
        
        return supports, resistances
    
    def _calculate_structure_quality(
        self,
        indicators: IndicatorValues,
        supports: List[float],
        resistances: List[float]
    ) -> int:
        """
        Calculate market structure quality score (0-100).
        
        High quality = clear trends, defined S/R, good volume
        """
        score = 50  # Base score
        
        # Clear S/R levels (+20)
        if len(supports) >= 2 and len(resistances) >= 2:
            score += 20
        elif len(supports) >= 1 or len(resistances) >= 1:
            score += 10
        
        # ADX trend clarity (+15)
        if indicators.adx > 25:
            score += 15
        elif indicators.adx > 20:
            score += 8
        
        # Volume confirmation (+15)
        if indicators.volume_ratio > 1.2:
            score += 15
        elif indicators.volume_ratio > 1.0:
            score += 8
        
        # BB position clarity (not in squeeze but not too wide)
        if 0.2 < indicators.bb_percent < 0.8:
            score += 0  # Middle = uncertain
        else:
            score += 10  # Near edges = clearer direction
        
        return min(100, score)
    
    def _calculate_entry_zone(
        self,
        current_price: float,
        supports: List[float],
        resistances: List[float],
        direction: str
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate optimal entry zone based on market structure.
        """
        if direction in ("bullish", "LONG"):
            # For longs, entry zone near support
            if supports:
                nearest_support = supports[0]
                zone_low = nearest_support
                zone_high = current_price
                return (zone_low, zone_high)
            else:
                # No support found, use 1% below current
                return (current_price * 0.99, current_price)
        else:
            # For shorts, entry zone near resistance
            if resistances:
                nearest_resistance = resistances[0]
                zone_low = current_price
                zone_high = nearest_resistance
                return (zone_low, zone_high)
            else:
                # No resistance found, use 1% above current
                return (current_price, current_price * 1.01)
    
    def _calculate_exit_zone(
        self,
        current_price: float,
        supports: List[float],
        resistances: List[float],
        direction: str,
        atr: float
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate optimal exit zone for take profit.
        """
        if direction in ("bullish", "LONG"):
            # For longs, exit near resistance
            if resistances:
                target = resistances[0]
                return (target * 0.99, target)
            else:
                # No resistance, use 2x ATR
                target = current_price + (atr * 2)
                return (target * 0.95, target)
        else:
            # For shorts, exit near support
            if supports:
                target = supports[0]
                return (target, target * 1.01)
            else:
                # No support, use 2x ATR below
                target = current_price - (atr * 2)
                return (target, target * 1.05)
    
    def _calculate_risk_reward(
        self,
        current_price: float,
        entry_zone: Optional[Tuple[float, float]],
        exit_zone: Optional[Tuple[float, float]],
        supports: List[float],
        direction: str
    ) -> float:
        """
        Calculate risk/reward ratio based on structure.
        """
        if not entry_zone or not exit_zone:
            return 1.0  # Default 1:1
        
        if direction in ("bullish", "LONG"):
            # Risk = entry to support (or 1%)
            stop_level = supports[0] if supports else current_price * 0.99
            risk = current_price - stop_level
            
            # Reward = entry to exit zone
            reward = exit_zone[0] - current_price
        else:
            # Risk = entry to resistance (or 1%)
            stop_level = current_price * 1.01  # Use 1% if no resistance for shorts
            risk = stop_level - current_price
            
            # Reward = current price to exit zone
            reward = current_price - exit_zone[1]
        
        if risk <= 0:
            return 0.0
        
        return max(0, reward / risk)
    
    def _get_ai_confirmation(
        self, 
        symbol: str, 
        df: pd.DataFrame,
        direction: str
    ) -> Tuple[Optional[bool], int]:
        """
        Get AI model confirmation for trade direction.
        
        Returns:
            Tuple of (confirms_direction, confidence) or (None, 0) if unavailable
        """
        if not self.config.get('use_ai_confirmation', True):
            return None, 0
        
        try:
            # Try Lag-Llama first
            from ml.models.lag_llama_predictor import get_lag_llama_predictor
            predictor = get_lag_llama_predictor()
            signal = predictor.get_trading_signal(df, symbol)
            
            if signal and signal.get('confidence', 0) > 0:
                ai_direction = signal['direction']
                confidence = int(signal['confidence'] * 100)
                
                expected = 'bullish' if direction in ('bullish', 'LONG') else 'bearish'
                confirms = ai_direction == expected
                
                return confirms, confidence
                
        except Exception:
            pass
        
        try:
            # Fallback to LSTM
            from ml.models.lstm_predictor import lstm_predictor
            prediction = lstm_predictor.predict(df)
            
            if prediction:
                ai_direction = prediction.direction.lower()
                confidence = int(prediction.confidence * 100)
                
                expected = 'bullish' if direction in ('bullish', 'LONG') else 'bearish'
                confirms = ai_direction == expected
                
                return confirms, confidence
                
        except Exception:
            pass
        
        return None, 0
    
    def _calculate_trade_quality(
        self, 
        condition: MarketCondition,
        direction: str
    ) -> int:
        """
        Calculate overall trade quality score (0-100).
        
        Factors:
        - Trend alignment: 30 points
        - Structure quality: 25 points
        - Risk/reward: 25 points
        - Regime suitability: 20 points
        """
        score = 0
        
        # 1. Trend alignment (30 points)
        if direction:
            expected = 'bullish' if direction in ('bullish', 'LONG') else 'bearish'
            if condition.trend_direction == expected:
                score += int(30 * (condition.trend_confidence / 100))
            elif condition.trend_direction == 'neutral':
                score += 10
            # Opposite direction = 0 points
        else:
            score += int(15 * (condition.trend_confidence / 100))
        
        # 2. Structure quality (25 points)
        score += int(25 * (condition.structure_quality / 100))
        
        # 3. Risk/reward ratio (25 points)
        min_rr = self.config.get('min_rr_ratio', 1.5)
        if condition.risk_reward_ratio >= min_rr * 2:
            score += 25
        elif condition.risk_reward_ratio >= min_rr:
            score += int(25 * (condition.risk_reward_ratio / (min_rr * 2)))
        else:
            score += int(10 * (condition.risk_reward_ratio / min_rr))
        
        # 4. Regime suitability (20 points)
        if direction in ('bullish', 'LONG'):
            if condition.regime == MarketRegime.TRENDING_UP:
                score += 20
            elif condition.regime == MarketRegime.RANGING:
                score += 10  # Range trading can work for longs at support
            else:
                score += 5
        else:
            if condition.regime == MarketRegime.TRENDING_DOWN:
                score += 20
            elif condition.regime == MarketRegime.RANGING:
                score += 10
            else:
                score += 5
        
        return min(100, score)
    
    def _make_decision(
        self, 
        condition: MarketCondition,
        direction: str
    ) -> Tuple[bool, str]:
        """
        Make final go/no-go decision based on analysis.
        
        Returns:
            Tuple of (should_enter, block_reason)
        """
        min_quality = self.config.get('min_trade_quality', 60)
        min_trend_conf = self.config.get('min_trend_confidence', 50)
        min_rr = self.config.get('min_rr_ratio', 1.5)
        
        # Check trade quality
        if condition.trade_quality_score < min_quality:
            return False, f"Quality too low ({condition.trade_quality_score} < {min_quality})"
        
        # Check trend confidence for trend trades
        if condition.regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            if condition.trend_confidence < min_trend_conf:
                return False, f"Trend confidence low ({condition.trend_confidence}% < {min_trend_conf}%)"
        
        # Check risk/reward
        if condition.risk_reward_ratio < min_rr:
            return False, f"R:R too low ({condition.risk_reward_ratio:.2f} < {min_rr})"
        
        # Check regime alignment with direction
        if direction in ('bullish', 'LONG'):
            if condition.regime == MarketRegime.TRENDING_DOWN:
                return False, "Market trending down (counter-trend long)"
        else:
            if condition.regime == MarketRegime.TRENDING_UP:
                return False, "Market trending up (counter-trend short)"
        
        # High volatility warning (allow but with caution)
        if condition.regime == MarketRegime.HIGH_VOLATILITY:
            condition.confidence_adjustment -= 10
            condition.analysis_reasons.append("High volatility - reduced confidence")
        
        return True, ""


# Singleton instance
_market_analyzer = None


def get_market_analyzer() -> MarketAnalyzer:
    """Get or create the market analyzer singleton."""
    global _market_analyzer
    if _market_analyzer is None:
        _market_analyzer = MarketAnalyzer()
    return _market_analyzer


# Export for direct import
market_analyzer = None


def initialize_market_analyzer():
    """Initialize market analyzer (call at startup)."""
    global market_analyzer
    market_analyzer = get_market_analyzer()
    logger.info("[MARKET] MarketAnalyzer initialized")
    return market_analyzer
