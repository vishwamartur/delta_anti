"""
Trading Signal Generator
Combines technical indicators to generate trade entry/exit signals.
Now with ML/AI integration for improved trade quality.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict
import config
from analysis.indicators import IndicatorValues

# ML Components for trade quality improvement
try:
    from ml.models.lstm_predictor import lstm_predictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    
try:
    from ml.sentiment.market_sentiment import sentiment_analyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

try:
    from ml.models.lag_llama_predictor import get_lag_llama_predictor
    LAG_LLAMA_AVAILABLE = True
except ImportError:
    LAG_LLAMA_AVAILABLE = False

# Range trading strategy for slow markets
try:
    from strategy.range_strategy import get_range_strategy, MarketRegime
    RANGE_STRATEGY_AVAILABLE = True
except ImportError:
    RANGE_STRATEGY_AVAILABLE = False

# Pre-trade market analysis
try:
    from analysis.market_analyzer import get_market_analyzer, MarketCondition
    MARKET_ANALYZER_AVAILABLE = True
except ImportError:
    MARKET_ANALYZER_AVAILABLE = False
    MarketCondition = None

# Smart Money Concepts
try:
    from analysis.smart_money import smart_money_analyzer
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False

logger = logging.getLogger(__name__)


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
    market_condition: Optional['MarketCondition'] = None
    trade_quality: int = 0
    
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
    Now integrates with TradeHistoryAnalyzer for adaptive parameters.
    """
    
    def __init__(self):
        self.config = config.SIGNAL_CONFIG
        self.rsi_config = config.INDICATOR_CONFIG['rsi']
        self.adaptive_config = config.ADAPTIVE_TRADING_CONFIG
        self.market_config = getattr(config, 'MARKET_ANALYSIS_CONFIG', {'enabled': True})
        self._analyzer = None  # Lazy load
        self._market_analyzer = None  # Lazy load
        
    @property
    def analyzer(self):
        """Lazy load trade analyzer to avoid circular imports"""
        if self._analyzer is None and self.adaptive_config.get('enabled', True):
            try:
                from strategy.trade_analyzer import analyzer
                self._analyzer = analyzer
                logger.info("[ADAPTIVE] TradeHistoryAnalyzer loaded")
            except ImportError as e:
                logger.warning(f"[ADAPTIVE] Could not load analyzer: {e}")
        return self._analyzer
    
    @property
    def market_analyzer(self):
        """Lazy load market analyzer to avoid circular imports"""
        if self._market_analyzer is None and MARKET_ANALYZER_AVAILABLE:
            if self.market_config.get('enabled', True):
                try:
                    self._market_analyzer = get_market_analyzer()
                    logger.info("[MARKET] MarketAnalyzer loaded")
                except Exception as e:
                    logger.warning(f"[MARKET] Could not load analyzer: {e}")
        return self._market_analyzer
        
    def _calculate_confidence(self, scores: List[int]) -> int:
        """Calculate overall confidence from individual scores."""
        if not scores:
            return 0
        return int(sum(scores) / len(scores))
    
    def _get_stop_loss(self, entry: float, atr: float, is_long: bool) -> float:
        """Calculate stop loss based on ATR with adaptive adjustment."""
        # Use adaptive multiplier if available
        multiplier = self.config['atr_multiplier_sl']
        if self.analyzer:
            params = self.analyzer.get_adaptive_parameters()
            multiplier = params.get('sl_atr_multiplier', multiplier)
        
        if is_long:
            return entry - (atr * multiplier)
        return entry + (atr * multiplier)
    
    def _get_take_profit(self, entry: float, atr: float, is_long: bool) -> float:
        """Calculate take profit based on ATR with adaptive adjustment."""
        # Use adaptive multiplier if available
        multiplier = self.config['atr_multiplier_tp']
        if self.analyzer:
            params = self.analyzer.get_adaptive_parameters()
            multiplier = params.get('tp_atr_multiplier', multiplier)
        
        if is_long:
            return entry + (atr * multiplier)
        return entry - (atr * multiplier)
    
    def _get_ml_prediction(self, symbol: str, df=None) -> Dict:
        """Get ML prediction for price direction and confidence.
        
        Uses Lag-Llama (foundation model) if available, falls back to LSTM.
        
        Returns:
            Dict with 'direction' ('bullish'/'bearish'), 'confidence' (0-100), 'change_pct'
        """
        result = {'direction': 'neutral', 'confidence': 0, 'change_pct': 0.0, 'model': 'none'}
        
        if df is None:
            return result
        
        # Try Lag-Llama first (more advanced foundation model)
        if LAG_LLAMA_AVAILABLE:
            try:
                lag_llama = get_lag_llama_predictor()
                signal = lag_llama.get_trading_signal(df, symbol)
                if signal and signal.get('confidence', 0) > 0:
                    result['direction'] = signal['direction']
                    result['confidence'] = int(signal['confidence'] * 100)
                    result['change_pct'] = signal.get('change_pct', 0)
                    result['model'] = 'lag-llama'
                    logger.info(f"[LAG-LLAMA] {symbol}: {result['direction']} "
                               f"({result['confidence']}%, {result['change_pct']:+.2f}%)")
                    return result
            except Exception as e:
                logger.debug(f"[LAG-LLAMA] Error, falling back to LSTM: {e}")
        
        # Fallback to LSTM predictor
        if ML_AVAILABLE:
            try:
                prediction = lstm_predictor.predict(df)
                if prediction:
                    result['direction'] = prediction.direction.lower()
                    result['confidence'] = int(prediction.confidence * 100)
                    result['change_pct'] = prediction.predicted_change_pct
                    result['model'] = 'lstm'
                    logger.info(f"[LSTM] {symbol}: {result['direction']} "
                               f"({result['confidence']}%, {result['change_pct']:+.2f}%)")
            except Exception as e:
                logger.debug(f"[LSTM] Prediction error: {e}")
            
        return result
    
    def _get_sentiment(self, symbol: str) -> Dict:
        """Get market sentiment for symbol.
        
        Returns:
            Dict with 'score' (-1 to 1), 'direction' ('bullish'/'bearish'/'neutral')
        """
        result = {'score': 0.0, 'direction': 'neutral', 'confidence': 50}
        
        if not SENTIMENT_AVAILABLE:
            return result
            
        try:
            sentiment_signal = sentiment_analyzer.get_sentiment_signal(symbol)
            if sentiment_signal:
                result['score'] = sentiment_signal.get('score', 0)
                result['direction'] = sentiment_signal.get('direction', 'neutral')
                result['confidence'] = sentiment_signal.get('strength', 50)
                logger.info(f"[SENTIMENT] {symbol}: {result['direction']} "
                           f"(score: {result['score']:.2f})")
        except Exception as e:
            logger.debug(f"[SENTIMENT] Analysis error: {e}")
            
        return result
    
    def _validate_with_ml(self, signal_direction: str, symbol: str, df=None) -> tuple:
        """Validate technical signal with ML predictions.
        
        Returns:
            (is_confirmed: bool, confidence_boost: int, reasons: list)
        """
        is_confirmed = True
        confidence_boost = 0
        reasons = []
        
        # Get ML prediction
        ml_pred = self._get_ml_prediction(symbol, df)
        
        if ml_pred['confidence'] > 50:
            # Strong ML prediction
            expected_dir = 'bullish' if 'LONG' in signal_direction.upper() else 'bearish'
            if ml_pred['direction'] == expected_dir:
                confidence_boost += 15
                reasons.append(f"ML confirms {expected_dir} ({ml_pred['confidence']}%)")
            elif ml_pred['direction'] != 'neutral':
                # ML disagrees with signal direction
                is_confirmed = False
                reasons.append(f"ML conflicts: predicts {ml_pred['direction']}")
        
        # Get sentiment
        sentiment = self._get_sentiment(symbol)
        
        if abs(sentiment['score']) > 0.2:
            expected_dir = 'bullish' if 'LONG' in signal_direction.upper() else 'bearish'
            if sentiment['direction'] == expected_dir:
                confidence_boost += 10
                reasons.append(f"Sentiment confirms {expected_dir}")
            elif sentiment['direction'] != 'neutral':
                # Sentiment disagrees
                confidence_boost -= 10
                reasons.append(f"Sentiment: {sentiment['direction']}")
        
        return is_confirmed, confidence_boost, reasons
    
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
                        indicators: IndicatorValues,
                        df: 'pd.DataFrame' = None) -> TradeSignal:
        """
        Generate trading signal based on technical indicators.
        
        Args:
            symbol: Trading symbol
            indicators: Calculated indicator values
            df: Optional OHLCV DataFrame for enhanced market analysis
            
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
            # No clear momentum signal - try range/scalping strategy for slow markets
            if RANGE_STRATEGY_AVAILABLE:
                range_strat = get_range_strategy()
                range_signal = range_strat.generate_signal(symbol, indicators, entry_price)
                
                if range_signal and range_signal.confidence >= self.config['min_confidence']:
                    # Convert range signal to TradeSignal
                    is_long = range_signal.direction == "LONG"
                    signal_type = SignalType.LONG if is_long else SignalType.SHORT
                    
                    logger.info(f"[RANGE] Using range strategy for slow market: "
                               f"{signal_type.value} (confidence={range_signal.confidence:.0f}%)")
                    
                    return TradeSignal(
                        symbol=symbol,
                        signal_type=signal_type,
                        strength=SignalStrength.MODERATE,
                        confidence=int(range_signal.confidence),
                        entry_price=range_signal.entry_price,
                        stop_loss=range_signal.stop_loss,
                        take_profit=range_signal.take_profit,
                        timestamp=datetime.now(),
                        reasons=[f"[RANGE] {range_signal.reason}"],
                        indicators=indicators
                    )
            
            # No trend or range signal
            return TradeSignal(
                symbol=symbol,
                signal_type=SignalType.NEUTRAL,
                strength=SignalStrength.WEAK,
                confidence=max(long_score, short_score),
                entry_price=entry_price,
                stop_loss=0,
                take_profit=0,
                timestamp=datetime.now(),
                reasons=["No clear signal (momentum or range)"],
                indicators=indicators
            )
        
        # === ADAPTIVE FILTER: Check if trade should be taken ===
        if self.analyzer and self.adaptive_config.get('enabled', True):
            direction = "LONG" if is_long else "SHORT"
            should_take, reason = self.analyzer.should_take_trade(symbol, direction, confidence)
            
            if not should_take:
                logger.info(f"[ADAPTIVE] BLOCKED {symbol} {direction}: {reason}")
                return TradeSignal(
                    symbol=symbol,
                    signal_type=SignalType.NEUTRAL,
                    strength=SignalStrength.WEAK,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=0,
                    take_profit=0,
                    timestamp=datetime.now(),
                    reasons=[f"BLOCKED: {reason}"],
                    indicators=indicators
                )
        
        # === ML VALIDATION: Confirm signal with AI predictions ===
        direction = "LONG" if is_long else "SHORT"
        ml_confirmed, ml_boost, ml_reasons = self._validate_with_ml(direction, symbol)
        
        if not ml_confirmed:
            # ML strongly disagrees with the signal
            logger.info(f"[ML] BLOCKED {symbol} {direction}: {ml_reasons}")
            return TradeSignal(
                symbol=symbol,
                signal_type=SignalType.NEUTRAL,
                strength=SignalStrength.WEAK,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=0,
                take_profit=0,
                timestamp=datetime.now(),
                reasons=[f"ML CONFLICT: {', '.join(ml_reasons)}"],
                indicators=indicators
            )
        
        # Apply ML confidence boost
        confidence = min(100, confidence + ml_boost)
        reasons.extend(ml_reasons)
        
        # === SMC VALIDATION: Check for Institutional Setups ===
        if SMC_AVAILABLE and df is not None:
            try:
                smc_analysis = smart_money_analyzer.analyze(df)
                
                # Check for SMC confirmations
                if is_long and smc_analysis.signal_direction == "bullish":
                    confidence = min(100, confidence + 15)
                    reasons.append(f"SMC: {smc_analysis.signal_reason}")
                elif not is_long and smc_analysis.signal_direction == "bearish":
                    confidence = min(100, confidence + 15)
                    reasons.append(f"SMC: {smc_analysis.signal_reason}")
                
                # Check for Liquidity Grabs (Strong Reversal Signal)
                for sweep in smc_analysis.liquidity_sweeps:
                    if is_long and sweep.direction == "bullish":
                         # Sweep of lows -> Bullish
                         confidence = min(100, confidence + 20)
                         reasons.append("SMC: Liquidity Grab (Lows swept)")
                    elif not is_long and sweep.direction == "bearish":
                         # Sweep of highs -> Bearish
                         confidence = min(100, confidence + 20)
                         reasons.append("SMC: Liquidity Grab (Highs swept)")
                         
            except Exception as e:
                logger.warning(f"[SMC] Analysis failed: {e}")
        
        # === PRE-TRADE MARKET ANALYSIS ===
        market_condition = None
        trade_quality = 0
        
        if df is not None and self.market_analyzer:
            direction = "LONG" if is_long else "SHORT"
            market_condition = self.market_analyzer.analyze_market(symbol, df, indicators, direction)
            trade_quality = market_condition.trade_quality_score
            
            if not market_condition.should_enter:
                logger.info(f"[MARKET] BLOCKED {symbol} {direction}: {market_condition.block_reason}")
                return TradeSignal(
                    symbol=symbol,
                    signal_type=SignalType.NEUTRAL,
                    strength=SignalStrength.WEAK,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=0,
                    take_profit=0,
                    timestamp=datetime.now(),
                    reasons=[f"MARKET: {market_condition.block_reason}"],
                    indicators=indicators,
                    market_condition=market_condition,
                    trade_quality=trade_quality
                )
            
            # Apply market analysis confidence adjustment
            confidence = min(100, confidence + market_condition.confidence_adjustment)
            reasons.extend(market_condition.analysis_reasons)
            
            # Use structure-based entry/exit if available
            if market_condition.support_levels and is_long:
                # Use nearest support for stop loss
                structure_sl = market_condition.support_levels[0] - (atr * 0.2)
                if structure_sl > 0:
                    stop_loss = structure_sl
                    reasons.append(f"SL at support ${structure_sl:,.2f}")
            
            if market_condition.resistance_levels and is_long:
                # Use nearest resistance for take profit
                structure_tp = market_condition.resistance_levels[0]
                if structure_tp > entry_price:
                    take_profit = structure_tp
                    reasons.append(f"TP at resistance ${structure_tp:,.2f}")
            
            if market_condition.resistance_levels and not is_long:
                # For shorts: resistance is stop, support is target
                structure_sl = market_condition.resistance_levels[0] + (atr * 0.2)
                stop_loss = structure_sl
            
            if market_condition.support_levels and not is_long:
                structure_tp = market_condition.support_levels[0]
                if structure_tp < entry_price:
                    take_profit = structure_tp
        else:
            # Calculate SL and TP without market analysis
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
            indicators=indicators,
            market_condition=market_condition,
            trade_quality=trade_quality
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
