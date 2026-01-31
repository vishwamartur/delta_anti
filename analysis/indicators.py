"""
Technical Analysis Indicators Module
Calculates various technical indicators for trading signals.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import config


@dataclass
class IndicatorValues:
    """Container for all calculated indicator values."""
    # Price
    price: float = 0.0
    
    # Moving Averages
    sma_20: float = 0.0
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    
    # RSI
    rsi: float = 50.0
    
    # MACD
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    
    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    bb_percent: float = 0.5  # Price position within bands (0-1)
    
    # Volatility
    atr: float = 0.0
    atr_percent: float = 0.0  # ATR as % of price
    
    # Momentum
    momentum: float = 0.0
    roc: float = 0.0  # Rate of Change
    
    # Volume
    volume: float = 0.0
    volume_sma: float = 0.0
    volume_ratio: float = 1.0  # Current volume / SMA volume
    
    # Trend
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    trend_strength: str = "neutral"  # "strong_up", "up", "neutral", "down", "strong_down"


class TechnicalIndicators:
    """
    Technical indicators calculator.
    All methods work on pandas DataFrames with OHLCV data.
    """
    
    def __init__(self):
        self.config = config.INDICATOR_CONFIG
    
    # ========== Moving Averages ==========
    
    def sma(self, series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()
    
    def ema(self, series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    # ========== RSI ==========
    
    def rsi(self, close: pd.Series, period: int = None) -> pd.Series:
        """
        Relative Strength Index (RSI).
        """
        period = period or self.config['rsi']['period']
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    # ========== MACD ==========
    
    def macd(self, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence.
        Returns: (macd_line, signal_line, histogram)
        """
        fast = self.config['macd']['fast']
        slow = self.config['macd']['slow']
        signal = self.config['macd']['signal']
        
        ema_fast = self.ema(close, fast)
        ema_slow = self.ema(close, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    # ========== Bollinger Bands ==========
    
    def bollinger_bands(self, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        Returns: (upper_band, middle_band, lower_band)
        """
        period = self.config['bollinger']['period']
        std_dev = self.config['bollinger']['std_dev']
        
        middle = self.sma(close, period)
        std = close.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def bb_percent(self, close: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
        """
        Bollinger Bands %B - Position of price within bands.
        0 = at lower band, 1 = at upper band
        """
        return (close - lower) / (upper - lower)
    
    def bb_width(self, upper: pd.Series, lower: pd.Series, middle: pd.Series) -> pd.Series:
        """Bollinger Band Width - Measures band spread."""
        return (upper - lower) / middle
    
    # ========== ATR (Average True Range) ==========
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = None) -> pd.Series:
        """Average True Range - Volatility indicator."""
        period = period or self.config['atr']['period']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    # ========== ADX (Average Directional Index) ==========
    
    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index - Trend strength indicator.
        Returns: (adx, plus_di, minus_di)
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        
        # Smoothed values
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    # ========== Momentum ==========
    
    def momentum(self, close: pd.Series, period: int = 10) -> pd.Series:
        """Momentum - Price change over period."""
        return close - close.shift(period)
    
    def roc(self, close: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change - Percentage change over period."""
        return ((close - close.shift(period)) / close.shift(period)) * 100
    
    # ========== Volume ==========
    
    def volume_sma(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume Simple Moving Average."""
        return self.sma(volume, period)
    
    # ========== Combined Calculation ==========
    
    def calculate_all(self, df: pd.DataFrame) -> Optional[IndicatorValues]:
        """
        Calculate all indicators and return latest values.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
        
        Returns:
            IndicatorValues with all latest indicator values
        """
        if df is None or len(df) < 50:  # Need minimum data
            return None
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Calculate all indicators
        # Moving Averages
        sma_20 = self.sma(close, 20)
        ema_9 = self.ema(close, self.config['ema']['short'])
        ema_21 = self.ema(close, self.config['ema']['medium'])
        ema_50 = self.ema(close, self.config['ema']['long'])
        
        # RSI
        rsi = self.rsi(close)
        
        # MACD
        macd_line, macd_signal, macd_hist = self.macd(close)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(close)
        bb_pct = self.bb_percent(close, bb_lower, bb_upper)
        bb_w = self.bb_width(bb_upper, bb_lower, bb_middle)
        
        # ATR
        atr = self.atr(high, low, close)
        
        # ADX
        adx, plus_di, minus_di = self.adx(high, low, close)
        
        # Momentum
        mom = self.momentum(close)
        roc = self.roc(close)
        
        # Volume
        vol_sma = self.volume_sma(volume)
        
        # Get latest values
        latest = IndicatorValues(
            price=close.iloc[-1],
            sma_20=sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else 0,
            ema_9=ema_9.iloc[-1] if not pd.isna(ema_9.iloc[-1]) else 0,
            ema_21=ema_21.iloc[-1] if not pd.isna(ema_21.iloc[-1]) else 0,
            ema_50=ema_50.iloc[-1] if not pd.isna(ema_50.iloc[-1]) else 0,
            rsi=rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            macd_line=macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0,
            macd_signal=macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0,
            macd_histogram=macd_hist.iloc[-1] if not pd.isna(macd_hist.iloc[-1]) else 0,
            bb_upper=bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else 0,
            bb_middle=bb_middle.iloc[-1] if not pd.isna(bb_middle.iloc[-1]) else 0,
            bb_lower=bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else 0,
            bb_width=bb_w.iloc[-1] if not pd.isna(bb_w.iloc[-1]) else 0,
            bb_percent=bb_pct.iloc[-1] if not pd.isna(bb_pct.iloc[-1]) else 0.5,
            atr=atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0,
            atr_percent=(atr.iloc[-1] / close.iloc[-1] * 100) if not pd.isna(atr.iloc[-1]) else 0,
            momentum=mom.iloc[-1] if not pd.isna(mom.iloc[-1]) else 0,
            roc=roc.iloc[-1] if not pd.isna(roc.iloc[-1]) else 0,
            volume=volume.iloc[-1] if not pd.isna(volume.iloc[-1]) else 0,
            volume_sma=vol_sma.iloc[-1] if not pd.isna(vol_sma.iloc[-1]) else 0,
            volume_ratio=(volume.iloc[-1] / vol_sma.iloc[-1]) if vol_sma.iloc[-1] > 0 else 1,
            adx=adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0,
            plus_di=plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 0,
            minus_di=minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 0
        )
        
        # Determine trend strength
        if latest.adx > 40:
            if latest.plus_di > latest.minus_di:
                latest.trend_strength = "strong_up"
            else:
                latest.trend_strength = "strong_down"
        elif latest.adx > 25:
            if latest.plus_di > latest.minus_di:
                latest.trend_strength = "up"
            else:
                latest.trend_strength = "down"
        else:
            latest.trend_strength = "neutral"
        
        return latest


# Singleton instance
indicators = TechnicalIndicators()
