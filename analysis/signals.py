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

# Multi-Timeframe Analysis
try:
    from analysis.multi_timeframe import get_multi_timeframe_analyzer
    MTF_AVAILABLE = True
except ImportError:
    MTF_AVAILABLE = False

# DQN Trading Agent
try:
    from ml.agents.dqn_trader import dqn_agent
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False

# Model Accuracy Tracking
try:
    from analysis.model_accuracy import get_accuracy_tracker
    ACCURACY_TRACKING_AVAILABLE = True
except ImportError:
    ACCURACY_TRACKING_AVAILABLE = False

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

        # Initialize Accuracy Tracker
        if ACCURACY_TRACKING_AVAILABLE:
            self.accuracy_tracker = get_accuracy_tracker()
        else:
            self.accuracy_tracker = None
        
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
        """Get ML prediction using ensemble voting (parallel LSTM + Lag-Llama).
        
        Runs both models and combines their predictions:
        - Both agree → high confidence (average + consensus bonus)
        - Disagree → use stronger model, reduce confidence
        - One model only → use as-is
        
        Returns:
            Dict with 'direction', 'confidence' (0-100), 'change_pct', 'model', 'ensemble'
        """
        result = {'direction': 'neutral', 'confidence': 0, 'change_pct': 0.0, 'model': 'none', 'ensemble': False}
        ensemble_cfg = getattr(config, 'ENSEMBLE_CONFIG', {})
        
        if df is None:
            return result
        
        predictions = []  # Collect predictions from all available models
        
        # === Model 1: Lag-Llama ===
        if LAG_LLAMA_AVAILABLE:
            try:
                lag_llama = get_lag_llama_predictor()
                signal = lag_llama.get_trading_signal(df, symbol)
                if signal and signal.get('confidence', 0) > 0:
                    # Apply accuracy weighting
                    weight = 1.0
                    if self.accuracy_tracker:
                        # Record prediction
                        self.accuracy_tracker.record_prediction(
                            f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}", 
                            'lag-llama', 
                            signal['direction']
                        )
                        weight = self.accuracy_tracker.get_weight('lag-llama')
                    
                    raw_conf = int(signal['confidence'] * 100)
                    weighted_conf = min(100, int(raw_conf * weight))
                    
                    predictions.append({
                        'direction': signal['direction'],
                        'confidence': weighted_conf,
                        'raw_confidence': raw_conf,
                        'change_pct': signal.get('change_pct', 0),
                        'model': 'lag-llama',
                        'weight': weight
                    })
                    logger.info(f"[ENSEMBLE:LAG-LLAMA] {symbol}: {signal['direction']} "
                               f"({weighted_conf}% [raw {raw_conf}% x {weight}x])")
            except Exception as e:
                logger.debug(f"[ENSEMBLE:LAG-LLAMA] Error: {e}")
        
        # === Model 2: LSTM ===
        if ML_AVAILABLE:
            try:
                prediction = lstm_predictor.predict(df)
                if prediction:
                    direction_map = {'up': 'bullish', 'down': 'bearish', 'neutral': 'neutral'}
                    lstm_dir = direction_map.get(prediction.direction.lower(), 'neutral')
                    
                    # Apply accuracy weighting
                    weight = 1.0
                    if self.accuracy_tracker:
                        # Record prediction
                        self.accuracy_tracker.record_prediction(
                            f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}", 
                            'lstm', 
                            lstm_dir
                        )
                        weight = self.accuracy_tracker.get_weight('lstm')
                    
                    raw_conf = int(prediction.confidence)
                    weighted_conf = min(100, int(raw_conf * weight))
                    
                    predictions.append({
                        'direction': lstm_dir,
                        'confidence': weighted_conf,
                        'raw_confidence': raw_conf,
                        'change_pct': prediction.predicted_change_pct,
                        'model': 'lstm',
                        'weight': weight
                    })
                    logger.info(f"[ENSEMBLE:LSTM] {symbol}: {lstm_dir} "
                               f"({weighted_conf}% [raw {raw_conf}% x {weight}x])")
            except Exception as e:
                logger.debug(f"[ENSEMBLE:LSTM] Error: {e}")
        
        # === Ensemble Decision ===
        if not predictions:
            return result
        
        if len(predictions) == 1:
            # Single model — use directly
            p = predictions[0]
            result['direction'] = p['direction']
            result['confidence'] = p['confidence']
            result['change_pct'] = p['change_pct']
            result['model'] = p['model']
            result['ensemble'] = False
        else:
            # Two models — ensemble voting
            p1, p2 = predictions[0], predictions[1]
            consensus_bonus = ensemble_cfg.get('consensus_bonus', 20)
            
            if p1['direction'] == p2['direction']:
                # CONSENSUS: Both models agree
                avg_conf = (p1['confidence'] + p2['confidence']) // 2
                result['direction'] = p1['direction']
                result['confidence'] = min(100, avg_conf + consensus_bonus)
                result['change_pct'] = (p1['change_pct'] + p2['change_pct']) / 2
                result['model'] = f"{p1['model']}+{p2['model']}"
                result['ensemble'] = True
                logger.info(f"[ENSEMBLE] ✅ CONSENSUS {symbol}: {result['direction']} "
                           f"({result['confidence']}% = avg {avg_conf}% + {consensus_bonus}% bonus)")
            else:
                # DISAGREEMENT: Use stronger model, penalize confidence
                stronger = p1 if p1['confidence'] > p2['confidence'] else p2
                weaker = p2 if stronger == p1 else p1
                penalty = min(15, abs(weaker['confidence'] - 30))  # Penalty based on weaker's conviction
                result['direction'] = stronger['direction']
                result['confidence'] = max(0, stronger['confidence'] - penalty)
                result['change_pct'] = stronger['change_pct']
                result['model'] = f"{stronger['model']}>{weaker['model']}"
                result['ensemble'] = True
                logger.info(f"[ENSEMBLE] ⚠️ SPLIT {symbol}: using {stronger['model']} "
                           f"({result['direction']} {result['confidence']}%, "
                           f"{weaker['model']} said {weaker['direction']})")
        
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
                # Lowercase to match expected format (bullish/bearish/neutral)
                result['direction'] = sentiment_signal.get('direction', 'neutral').lower()
                result['confidence'] = sentiment_signal.get('strength', 50)
                logger.info(f"[SENTIMENT] {symbol}: {result['direction']} "
                           f"(score: {result['score']:.2f})")
        except Exception as e:
            logger.debug(f"[SENTIMENT] Analysis error: {e}")
            
        return result
    
    def _build_dqn_state(self, indicators: 'IndicatorValues') -> 'np.ndarray':
        """Build 50-dim state vector for DQN from indicator values."""
        import numpy as np
        state = np.zeros(50, dtype=np.float32)
        
        try:
            price = indicators.price or 0
            
            # Price features (0-4)
            state[0] = price / 100000 if price > 0 else 0  # Normalized BTC price
            state[1] = getattr(indicators, 'price_change_pct', 0) or 0
            state[2] = (indicators.atr / price * 100) if (indicators.atr and price) else 0
            state[3] = ((price - indicators.sma_20) / indicators.sma_20 * 100) if indicators.sma_20 else 0
            state[4] = ((price - indicators.ema_21) / indicators.ema_21 * 100) if indicators.ema_21 else 0
            
            # Oscillators (5-9)
            state[5] = (indicators.rsi / 100) if indicators.rsi else 0.5
            state[6] = (indicators.macd_line / price * 1000) if (indicators.macd_line and price) else 0
            state[7] = (indicators.macd_signal / price * 1000) if (indicators.macd_signal and price) else 0
            state[8] = (indicators.macd_histogram / price * 1000) if (indicators.macd_histogram and price) else 0
            state[9] = (indicators.adx / 100) if indicators.adx else 0
            
            # Bollinger Bands (10-12)
            if indicators.bb_upper and indicators.bb_lower and price:
                bb_range = indicators.bb_upper - indicators.bb_lower
                state[10] = (price - indicators.bb_lower) / bb_range if bb_range > 0 else 0.5  # %B
                state[11] = bb_range / price  # Band width
            state[12] = (indicators.bb_middle / price) if (indicators.bb_middle and price) else 1.0
            
            # EMA features (13-16)
            if indicators.ema_9 and indicators.ema_21:
                state[13] = 1.0 if indicators.ema_9 > indicators.ema_21 else -1.0  # Cross
                state[14] = (indicators.ema_9 - indicators.ema_21) / indicators.ema_21 * 100  # Spread
            ema_50 = getattr(indicators, 'ema_50', None)
            if ema_50 and indicators.ema_21:
                state[15] = 1.0 if indicators.ema_21 > ema_50 else -1.0
                state[16] = (indicators.ema_21 - ema_50) / ema_50 * 100
            
            # Volume (17-18)
            state[17] = min(indicators.volume_ratio / 3.0, 1.0) if indicators.volume_ratio else 0.5
            state[18] = 1.0 if (indicators.volume_ratio and indicators.volume_ratio > 1.5) else 0.0
            
            # DI+/DI- (19-20)
            state[19] = (indicators.plus_di / 100) if indicators.plus_di else 0
            state[20] = (indicators.minus_di / 100) if indicators.minus_di else 0
            
        except Exception as e:
            logger.debug(f"[DQN] State build error: {e}")
        
        return state
    
    def _get_dqn_action(self, indicators: 'IndicatorValues') -> dict:
        """Get DQN agent's recommended action and confidence.
        
        Returns:
            {'action': int, 'action_name': str, 'confidence': float, 'probs': array}
        """
        result = {'action': 0, 'action_name': 'HOLD', 'confidence': 0.33, 'probs': None}
        
        if not DQN_AVAILABLE:
            return result
        
        try:
            state = self._build_dqn_state(indicators)
            action = dqn_agent.get_action(state, training=False)  # Greedy
            probs = dqn_agent.get_action_probs(state)
            
            result['action'] = action
            result['action_name'] = dqn_agent.get_action_name(action)
            result['confidence'] = float(probs[action]) if probs is not None else 0.33
            result['probs'] = probs
            result['state'] = state  # Saved for experience replay
            
            logger.info(f"[DQN] Action: {result['action_name']} "
                       f"(confidence={result['confidence']:.2f}, "
                       f"probs=[H:{probs[0]:.2f} B:{probs[1]:.2f} S:{probs[2]:.2f}])")
        except Exception as e:
            logger.debug(f"[DQN] Action error: {e}")
        
        return result
    
    def _validate_with_ml(self, signal_direction: str, symbol: str, 
                          df=None, df_htf=None, indicators=None) -> tuple:
        """Validate technical signal with ML ensemble + HTF trend + sentiment + DQN.
        
        Returns:
            (is_confirmed: bool, confidence_boost: int, reasons: list)
        """
        is_confirmed = True
        confidence_boost = 0
        reasons = []
        ml_cfg = getattr(config, 'ML_VALIDATION_CONFIG', {})
        
        # === Multi-Timeframe Filter ===
        mtf_config = getattr(config, 'MULTI_TIMEFRAME_CONFIG', {})
        if MTF_AVAILABLE and mtf_config.get('enabled', True) and df_htf is not None:
            try:
                mtf_analyzer = get_multi_timeframe_analyzer()
                htf_trend = mtf_analyzer.analyze_trend(df_htf)
                confirms, htf_boost, htf_reason = mtf_analyzer.confirms_direction(
                    htf_trend, signal_direction
                )
                
                if not confirms:
                    is_confirmed = False
                    reasons.append(f"HTF BLOCKED: {htf_reason}")
                    return is_confirmed, confidence_boost, reasons
                
                confidence_boost += htf_boost
                if htf_boost != 0:
                    reasons.append(htf_reason)
            except Exception as e:
                logger.debug(f"[HTF] Analysis error: {e}")
        
        # === Ensemble ML Prediction ===
        ml_pred = self._get_ml_prediction(symbol, df)
        ml_threshold = ml_cfg.get('ml_confidence_threshold', 60)
        ml_boost = ml_cfg.get('ml_confirm_boost', 15)
        
        if ml_pred['confidence'] > ml_threshold:
            expected_dir = 'bullish' if 'LONG' in signal_direction.upper() else 'bearish'
            if ml_pred['direction'] == expected_dir:
                # Extra boost for ensemble consensus
                boost = ml_boost + (5 if ml_pred.get('ensemble') else 0)
                confidence_boost += boost
                model_label = f"Ensemble" if ml_pred.get('ensemble') else ml_pred['model']
                reasons.append(f"{model_label} confirms {expected_dir} ({ml_pred['confidence']}%)")
            elif ml_pred['direction'] != 'neutral':
                if ml_cfg.get('ml_block_on_disagree', True):
                    is_confirmed = False
                    reasons.append(f"ML REJECTS: {ml_pred['model']} predicts {ml_pred['direction']}")
        
        # === Sentiment Analysis ===
        sentiment = self._get_sentiment(symbol)
        sent_threshold = ml_cfg.get('sentiment_score_threshold', 0.3)
        sent_boost = ml_cfg.get('sentiment_confirm_boost', 10)
        sent_penalty = ml_cfg.get('sentiment_penalty', -10)
        
        # Apply accuracy weighting to sentiment
        if self.accuracy_tracker:
            weight = self.accuracy_tracker.get_weight('sentiment')
            sent_boost = int(sent_boost * weight)
            # Record prediction
            if sentiment['direction'] != 'neutral':
                self.accuracy_tracker.record_prediction(
                    f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}", 
                    'sentiment', 
                    sentiment['direction']
                )

        
        if abs(sentiment['score']) > sent_threshold:
            expected_dir = 'bullish' if 'LONG' in signal_direction.upper() else 'bearish'
            if sentiment['direction'] == expected_dir:
                confidence_boost += sent_boost
                reasons.append(f"Sentiment supports {expected_dir}")
            elif sentiment['direction'] != 'neutral':
                confidence_boost += sent_penalty
                reasons.append(f"Sentiment opposes: {sentiment['direction']}")
        
        # === DQN Agent Validation ===
        dqn_cfg = getattr(config, 'DQN_INTEGRATION_CONFIG', {})
        if DQN_AVAILABLE and dqn_cfg.get('enabled', False) and indicators is not None:
            try:
                dqn_result = self._get_dqn_action(indicators)
                action = dqn_result['action']
                
                # Map signal direction to expected DQN action
                expected_action = 1 if 'LONG' in signal_direction.upper() else 2  # BUY=1, SELL=2
                
                if action == expected_action:
                    # DQN agrees with signal
                    boost = dqn_cfg.get('dqn_confirm_boost', 10)
                    confidence_boost += boost
                    reasons.append(f"DQN supports {dqn_result['action_name']} "
                                 f"({dqn_result['confidence']:.0%})")
                elif action == 0:  # HOLD
                    penalty = dqn_cfg.get('dqn_hold_penalty', -5)
                    confidence_boost += penalty
                    reasons.append(f"DQN suggests HOLD ({dqn_result['confidence']:.0%})")
                else:
                    # DQN opposes (wants opposite direction)
                    penalty = dqn_cfg.get('dqn_oppose_penalty', -15)
                    confidence_boost += penalty
                    reasons.append(f"DQN opposes: wants {dqn_result['action_name']} "
                                 f"({dqn_result['confidence']:.0%})")
                
                # Store DQN state for experience replay (used by run_system.py)
                self._last_dqn_state = dqn_result.get('state')
                self._last_dqn_action = action
            except Exception as e:
                logger.debug(f"[DQN] Validation error: {e}")
        
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
                        df: 'pd.DataFrame' = None,
                        df_htf: 'pd.DataFrame' = None) -> TradeSignal:
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
        ml_confirmed, ml_boost, ml_reasons = self._validate_with_ml(direction, symbol, df, df_htf, indicators=indicators)
        
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
        
        # Default SL/TP based on ATR (may be overridden by market analysis)
        stop_loss = self._get_stop_loss(entry_price, atr, is_long)
        take_profit = self._get_take_profit(entry_price, atr, is_long)
        
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
        # (SL/TP defaults already set above, no else needed)
        
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
