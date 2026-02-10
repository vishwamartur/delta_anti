"""
Multi-Timeframe Trend Analysis Module

Analyzes higher timeframe (e.g., 1h) to confirm trade direction.
Only allows entries aligned with the dominant trend.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd
import numpy as np
import config

logger = logging.getLogger(__name__)


@dataclass
class HTFTrend:
    """Higher Timeframe trend analysis result."""
    direction: str          # 'bullish', 'bearish', 'neutral'
    strength: int           # 0-100
    ema_aligned: bool       # True if EMA 9 > 21 > 50 (bullish) or reversed
    adx_value: float        # ADX trend strength
    rsi_value: float        # RSI on higher timeframe
    reason: str             # Human-readable reason


class MultiTimeframeAnalyzer:
    """
    Analyzes higher timeframe data to determine dominant trend.
    
    Uses:
    - EMA 9/21/50 alignment for trend direction
    - ADX for trend strength
    - RSI for momentum confirmation
    """
    
    def __init__(self):
        self.config = getattr(config, 'MULTI_TIMEFRAME_CONFIG', {})
        self.ema_short = 9
        self.ema_medium = 21
        self.ema_long = 50
    
    def analyze_trend(self, df: pd.DataFrame) -> HTFTrend:
        """
        Analyze higher timeframe data for trend direction.
        
        Args:
            df: OHLCV DataFrame (e.g., 1h candles)
            
        Returns:
            HTFTrend with direction, strength, and supporting data
        """
        if df is None or len(df) < self.ema_long + 5:
            return HTFTrend(
                direction='neutral', strength=0, ema_aligned=False,
                adx_value=0, rsi_value=50, reason='Insufficient HTF data'
            )
        
        close = df['close']
        
        # Calculate EMAs
        ema_9 = close.ewm(span=self.ema_short, adjust=False).mean()
        ema_21 = close.ewm(span=self.ema_medium, adjust=False).mean()
        ema_50 = close.ewm(span=self.ema_long, adjust=False).mean()
        
        # Latest values
        e9 = ema_9.iloc[-1]
        e21 = ema_21.iloc[-1]
        e50 = ema_50.iloc[-1]
        price = close.iloc[-1]
        
        # EMA alignment check
        bullish_aligned = (price > e9 > e21 > e50)
        bearish_aligned = (price < e9 < e21 < e50)
        
        # Calculate RSI on HTF
        rsi = self._calculate_rsi(close, 14)
        
        # Calculate ADX on HTF
        adx = self._calculate_adx(df, 14)
        
        # Determine trend direction and strength
        direction = 'neutral'
        strength = 0
        reasons = []
        
        if bullish_aligned:
            direction = 'bullish'
            strength += 40
            reasons.append('EMA 9>21>50 aligned bullish')
        elif bearish_aligned:
            direction = 'bearish'
            strength += 40
            reasons.append('EMA 9<21<50 aligned bearish')
        else:
            # Partial alignment
            if e9 > e21:
                direction = 'bullish'
                strength += 20
                reasons.append('Short-term EMA bullish')
            elif e9 < e21:
                direction = 'bearish'
                strength += 20
                reasons.append('Short-term EMA bearish')
        
        # ADX strength bonus
        if adx > 25:
            strength += 20
            reasons.append(f'Strong trend (ADX={adx:.0f})')
        elif adx > 15:
            strength += 10
            reasons.append(f'Moderate trend (ADX={adx:.0f})')
        
        # RSI confirmation
        if direction == 'bullish' and rsi > 50:
            strength += 15
            reasons.append(f'RSI confirms bullish ({rsi:.0f})')
        elif direction == 'bearish' and rsi < 50:
            strength += 15
            reasons.append(f'RSI confirms bearish ({rsi:.0f})')
        elif direction == 'bullish' and rsi > 70:
            strength -= 10  # Overbought, caution
            reasons.append(f'RSI overbought ({rsi:.0f})')
        elif direction == 'bearish' and rsi < 30:
            strength -= 10  # Oversold, caution
            reasons.append(f'RSI oversold ({rsi:.0f})')
        
        # Price vs EMAs bonus
        price_above_all = price > e9 and price > e21 and price > e50
        price_below_all = price < e9 and price < e21 and price < e50
        
        if price_above_all and direction == 'bullish':
            strength += 15
        elif price_below_all and direction == 'bearish':
            strength += 15
        
        strength = max(0, min(100, strength))
        
        trend = HTFTrend(
            direction=direction,
            strength=strength,
            ema_aligned=(bullish_aligned or bearish_aligned),
            adx_value=adx,
            rsi_value=rsi,
            reason='; '.join(reasons)
        )
        
        logger.info(f"[HTF] Trend: {trend.direction} (strength={trend.strength}, "
                    f"EMA aligned={trend.ema_aligned}, ADX={trend.adx_value:.1f})")
        
        return trend
    
    def confirms_direction(self, htf_trend: HTFTrend, signal_direction: str) -> tuple:
        """
        Check if higher timeframe trend confirms the signal direction.
        
        Args:
            htf_trend: Higher timeframe trend analysis
            signal_direction: 'LONG' or 'SHORT'
            
        Returns:
            (confirms: bool, boost: int, reason: str)
        """
        expected = 'bullish' if 'LONG' in signal_direction.upper() else 'bearish'
        htf_boost = self.config.get('htf_trend_boost', 10)
        
        if htf_trend.direction == expected:
            if htf_trend.ema_aligned:
                return True, htf_boost, f"HTF trend strongly confirms {expected}"
            else:
                return True, htf_boost // 2, f"HTF trend partially confirms {expected}"
        elif htf_trend.direction == 'neutral':
            return True, 0, "HTF trend neutral — no filter applied"
        else:
            # HTF disagrees
            block = self.config.get('htf_block_penalty', True)
            if block and htf_trend.strength >= 50:
                return False, -htf_boost, f"HTF trend OPPOSES signal ({htf_trend.direction}, strength={htf_trend.strength})"
            else:
                return True, -5, f"Weak HTF opposition ({htf_trend.direction}, strength={htf_trend.strength})"
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        
        # Smoothed averages
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        adx = dx.rolling(window=period).mean()
        
        return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0


# Singleton
_analyzer = None

def get_multi_timeframe_analyzer() -> MultiTimeframeAnalyzer:
    """Get or create the multi-timeframe analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = MultiTimeframeAnalyzer()
    return _analyzer
