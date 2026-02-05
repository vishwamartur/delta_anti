"""
Institutional Trading Strategy (Smart Money Concepts) Module.

This module implements institutional trading concepts to identify high-probability
trade setups used by "Smart Money" or institutional algorithms.

Key Concepts:
1. Order Blocks (OB): Candles where institutions placed large orders before a strong move.
2. Fair Value Gaps (FVG): Imbalances in price action leaving unfilled orders.
3. Liquidity Grabs: Sweeps of swing highs/lows to trap retail traders.
4. Market Structure Shift (MSS): Confirmation of trend reversals.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class KeyLevel:
    """Represents a key price level (OB, FVG, Swing Point)."""
    price_high: float
    price_low: float
    type: str  # "OB", "FVG", "LIQ", "MSS"
    direction: str  # "bullish", "bearish"
    timestamp: datetime
    strength: int = 0  # 0-100 score
    is_mitigated: bool = False
    
    @property
    def mid_price(self) -> float:
        return (self.price_high + self.price_low) / 2

@dataclass
class SMCAnalysis:
    """Container for SMC analysis results."""
    order_blocks: List[KeyLevel] = field(default_factory=list)
    fvgs: List[KeyLevel] = field(default_factory=list)
    liquidity_sweeps: List[KeyLevel] = field(default_factory=list)
    market_structure: str = "neutral"  # "bullish", "bearish", "neutral"
    
    # Combined signal
    signal_direction: str = "neutral"
    signal_confidence: int = 0
    signal_reason: str = ""
    
    entry_zone: Optional[Tuple[float, float]] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None


class SmartMoneyAnalyzer:
    """
    Analyzer for Smart Money Concepts (SMC).
    Identifies institutional footprints in price action.
    """
    
    def __init__(self):
        self.min_body_ratio = 0.5  # Candle body must be at least 50% of range for momentum
        self.fvg_threshold = 0.001  # Minimum FVG size (0.1%)
        
    def analyze(self, df: pd.DataFrame) -> SMCAnalysis:
        """
        Perform full SMC analysis on DataFrame.
        """
        if df is None or len(df) < 50:
            return SMCAnalysis()
            
        analysis = SMCAnalysis()
        
        # 1. Detect Order Blocks
        analysis.order_blocks = self._find_order_blocks(df)
        
        # 2. Detect Fair Value Gaps
        analysis.fvgs = self._find_fair_value_gaps(df)
        
        # 3. Detect Liquidity Grabs
        analysis.liquidity_sweeps = self._find_liquidity_sweeps(df)
        
        # 4. Analyze Market Structure
        analysis.market_structure = self._analyze_structure(df)
        
        # 5. Generate Signal
        self._generate_smc_signal(df, analysis)
        
        return analysis
    
    def _find_order_blocks(self, df: pd.DataFrame, lookback: int = 50) -> List[KeyLevel]:
        """
        Identify potential Order Blocks.
        Bullish OB: Last bearish candle before a strong bullish move that breaks structure/takes liquidity.
        Bearish OB: Last bullish candle before a strong bearish move.
        """
        obs = []
        recent_df = df.iloc[-lookback:]
        
        # We need future context to confirm the move, so we iterate up to len-3
        # In a real-time scenario, we look back from current to find PAST OBs
        
        for i in range(len(recent_df) - 3):
            curr_idx = recent_df.index[i]
            next_idx = recent_df.index[i+1]
            next_next_idx = recent_df.index[i+2]
            
            curr_candle = recent_df.iloc[i]
            next_candle = recent_df.iloc[i+1]
            next_move = recent_df.iloc[i+1:i+4] # Next 3 candles
            
            # --- Bullish OB Pattern ---
            # 1. Current candle is bearish (Red)
            is_bearish_candle = curr_candle['close'] < curr_candle['open']
            
            if is_bearish_candle:
                # 2. Subsequent move is strong bullish (engulfing or strong expansion)
                move_high = next_move['high'].max()
                move_close = next_move['close'].iloc[-1]
                
                # Check for strong break above current candle high
                if move_close > curr_candle['high']:
                    # Calculate strength based on the move magnitude
                    move_range = (move_high - curr_candle['low']) / curr_candle['low']
                    if move_range > 0.01: # > 1% move
                        
                        # Check if OB is mitigated (price came back to it subsequently)
                        # Look at price AFTER the initial move away
                        future_candles = recent_df.iloc[i+4:]
                        is_mitigated = False
                        if not future_candles.empty:
                            # Mitigated if low triggers the OB range
                            min_future = future_candles['low'].min()
                            if min_future <= curr_candle['high']: 
                                is_mitigated = min_future < curr_candle['low'] # Invalidated if breaks low, mitigated if touches
                        
                        if not is_mitigated:
                            obs.append(KeyLevel(
                                price_high=curr_candle['high'],
                                price_low=curr_candle['low'],
                                type="OB",
                                direction="bullish",
                                timestamp=curr_idx,
                                strength=int(min(100, move_range * 5000)), # Scaled score
                                is_mitigated=False
                            ))
                            
            # --- Bearish OB Pattern ---
            # 1. Current candle is bullish (Green)
            is_bullish_candle = curr_candle['close'] > curr_candle['open']
            
            if is_bullish_candle:
                # 2. Subsequent move is strong bearish
                move_low = next_move['low'].min()
                move_close = next_move['close'].iloc[-1]
                
                # Check for strong break below current candle low
                if move_close < curr_candle['low']:
                    move_range = (curr_candle['high'] - move_low) / curr_candle['high']
                    if move_range > 0.01:
                        
                        future_candles = recent_df.iloc[i+4:]
                        is_mitigated = False
                        if not future_candles.empty:
                            max_future = future_candles['high'].max()
                            if max_future >= curr_candle['low']:
                                is_mitigated = max_future > curr_candle['high']
                                
                        if not is_mitigated:
                            obs.append(KeyLevel(
                                price_high=curr_candle['high'],
                                price_low=curr_candle['low'],
                                type="OB",
                                direction="bearish",
                                timestamp=curr_idx,
                                strength=int(min(100, move_range * 5000)),
                                is_mitigated=False
                            ))
        
        # Filter: Return only recent, unmitigated or high quality OBs
        # Sort by timestamp (most recent first)
        obs.sort(key=lambda x: x.timestamp, reverse=True)
        return obs[:5]
    
    def _find_fair_value_gaps(self, df: pd.DataFrame, lookback: int = 50) -> List[KeyLevel]:
        """
        Identify Fair Value Gaps (FVG) / Imbalances.
        Bullish FVG: Low of candle 3 is above High of candle 1.
        Bearish FVG: High of candle 3 is below Low of candle 1.
        """
        fvgs = []
        recent_df = df.iloc[-lookback:]
        
        for i in range(len(recent_df) - 2):
            idx1 = recent_df.index[i]
            
            candle1 = recent_df.iloc[i]
            # candle2 = recent_df.iloc[i+1] # The gap candle
            candle3 = recent_df.iloc[i+2]
            
            # Bullish FVG
            # Gap between Candle 1 High and Candle 3 Low
            if candle3['low'] > candle1['high']:
                gap_size = (candle3['low'] - candle1['high']) / candle1['high']
                
                if gap_size > self.fvg_threshold:
                    # Check mitigation
                    future_candles = recent_df.iloc[i+3:]
                    is_mitigated = False
                    if not future_candles.empty:
                        # Mitigated if price fills the gap
                        if future_candles['low'].min() <= candle1['high']:
                            is_mitigated = True
                    
                    if not is_mitigated:
                        fvgs.append(KeyLevel(
                            price_high=candle3['low'],
                            price_low=candle1['high'],
                            type="FVG",
                            direction="bullish",
                            timestamp=idx1,
                            strength=int(min(100, gap_size * 10000)),
                            is_mitigated=False
                        ))
            
            # Bearish FVG
            # Gap between Candle 1 Low and Candle 3 High
            elif candle3['high'] < candle1['low']:
                gap_size = (candle1['low'] - candle3['high']) / candle1['low']
                
                if gap_size > self.fvg_threshold:
                    future_candles = recent_df.iloc[i+3:]
                    is_mitigated = False
                    if not future_candles.empty:
                        if future_candles['high'].max() >= candle1['low']:
                            is_mitigated = True
                            
                    if not is_mitigated:
                        fvgs.append(KeyLevel(
                            price_high=candle1['low'],
                            price_low=candle3['high'],
                            type="FVG",
                            direction="bearish",
                            timestamp=idx1,
                            strength=int(min(100, gap_size * 10000)),
                            is_mitigated=False
                        ))
                        
        fvgs.sort(key=lambda x: x.timestamp, reverse=True)
        return fvgs[:5]
    
    def _find_liquidity_sweeps(self, df: pd.DataFrame, lookback: int = 20) -> List[KeyLevel]:
        """
        Detect liquidity sweeps (Swing failure patterns).
        Price breaks a recent swing high/low but closes inside the range (wick only).
        """
        sweeps = []
        recent_df = df.iloc[-lookback:].copy()
        
        # Identify swing points
        current_candle = recent_df.iloc[-1]
        
        # Check vs previous swing highs
        # Find local maxima in previous candles (excluding current)
        window = 5
        for i in range(window, len(recent_df)-1):
            is_swing_high = recent_df['high'].iloc[i] == recent_df['high'].iloc[i-window:i+window+1].max()
            
            if is_swing_high:
                swing_high = recent_df['high'].iloc[i]
                
                # Check if current candle swept this high
                # High > Swing High AND Close < Swing High
                if current_candle['high'] > swing_high and current_candle['close'] < swing_high:
                    sweeps.append(KeyLevel(
                        price_high=current_candle['high'],
                        price_low=swing_high,
                        type="LIQ_GRAB",
                        direction="bearish", # Swept high -> bearish reversal signal
                        timestamp=recent_df.index[-1],
                        strength=80,
                        is_mitigated=False
                    ))
                    
        # Check vs previous swing lows
        for i in range(window, len(recent_df)-1):
            is_swing_low = recent_df['low'].iloc[i] == recent_df['low'].iloc[i-window:i+window+1].min()
            
            if is_swing_low:
                swing_low = recent_df['low'].iloc[i]
                
                # Check if current candle swept this low
                # Low < Swing Low AND Close > Swing Low
                if current_candle['low'] < swing_low and current_candle['close'] > swing_low:
                    sweeps.append(KeyLevel(
                        price_high=swing_low,
                        price_low=current_candle['low'],
                        type="LIQ_GRAB",
                        direction="bullish", # Swept low -> bullish reversal signal
                        timestamp=recent_df.index[-1],
                        strength=80,
                        is_mitigated=False
                    ))
                    
        return sweeps
    
    def _analyze_structure(self, df: pd.DataFrame) -> str:
        """
        Identify market structure (HH/HL or LH/LL).
        """
        if len(df) < 20: return "neutral"
        
        # Simple structure based on comparison of last two significant swing points
        # Implementation simplified to recent price action vs MA
        ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
        
        if df['close'].iloc[-1] > ema_20 > ema_50:
            return "bullish"
        elif df['close'].iloc[-1] < ema_20 < ema_50:
            return "bearish"
        else:
            return "neutral"
            
    def _generate_smc_signal(self, df: pd.DataFrame, analysis: SMCAnalysis):
        """
        Combine factors into a trade signal.
        Logic:
        - Price dipping into Bullish OB or FVG = BUY
        - Price rising into Bearish OB or FVG = SELL
        - Liquidity Sweep = Reversal Signal
        """
        current_price = df['close'].iloc[-1]
        
        score_bullish = 0
        score_bearish = 0
        reasons = []
        
        # 1. Zone Proximity
        
        # Check Bullish Zones (Support)
        # Sort by proximity
        valid_bullish_zones = [z for z in analysis.order_blocks + analysis.fvgs if z.direction == "bullish"]
        for zone in valid_bullish_zones:
            # If price is inside or just above bullish zone
            dist_pct = (current_price - zone.price_high) / current_price
            if -0.005 < dist_pct < 0.01: # Within 0.5% below or 1% above
                score_bullish += 30
                reasons.append(f"In bullish {zone.type} ({zone.price_high:.0f}-{zone.price_low:.0f})")
                analysis.entry_zone = (zone.price_low, zone.price_high)
                analysis.stop_loss = zone.price_low * 0.995 # SL below zone
                break # Only count nearest
                
        # Check Bearish Zones (Resistance)
        valid_bearish_zones = [z for z in analysis.order_blocks + analysis.fvgs if z.direction == "bearish"]
        for zone in valid_bearish_zones:
            dist_pct = (zone.price_low - current_price) / current_price
            if -0.005 < dist_pct < 0.01:
                score_bearish += 30
                reasons.append(f"In bearish {zone.type} ({zone.price_low:.0f}-{zone.price_high:.0f})")
                analysis.entry_zone = (zone.price_low, zone.price_high)
                # For bearish, SL above zone
                analysis.stop_loss = zone.price_high * 1.005
                break
        
        # 2. Liquidity Grabs (Immediate Reversal)
        for sweep in analysis.liquidity_sweeps:
            if sweep.direction == "bullish":
                score_bullish += 50
                reasons.append("Bullish Liquidity Grab (SSL swept)")
            elif sweep.direction == "bearish":
                score_bearish += 50
                reasons.append("Bearish Liquidity Grab (BSL swept)")
        
        # 3. Market Structure Confirmation
        if analysis.market_structure == "bullish":
            score_bullish += 10
        elif analysis.market_structure == "bearish":
            score_bearish += 10
            
        # Determine Signal
        if score_bullish > score_bearish and score_bullish >= 30:
            analysis.signal_direction = "bullish"
            analysis.signal_confidence = min(95, score_bullish + 30) # Base confidence
            analysis.signal_reason = ", ".join(reasons)
        elif score_bearish > score_bullish and score_bearish >= 30:
            analysis.signal_direction = "bearish"
            analysis.signal_confidence = min(95, score_bearish + 30)
            analysis.signal_reason = ", ".join(reasons)
        else:
            analysis.signal_direction = "neutral"
            analysis.signal_confidence = 0
            
# Singleton
smart_money_analyzer = SmartMoneyAnalyzer()
