"""
Model Accuracy Tracker
Tracks ML model prediction accuracy over a rolling window and provides
dynamic weight multipliers for ensemble decisions.
"""
import json
import os
import logging
from datetime import datetime
from collections import deque
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Default config
DEFAULT_CONFIG = {
    'enabled': True,
    'window_size': 50,       # Rolling window of trades to evaluate
    'min_trades': 10,        # Minimum trades before adjusting weights
    'base_weight': 1.0,      # Default weight multiplier
    'max_weight': 1.5,       # Max boost for high-accuracy models
    'min_weight': 0.3,       # Min weight for poor-accuracy models
    'save_path': 'models/model_accuracy.json',
}


class ModelAccuracyTracker:
    """
    Tracks each ML model's directional prediction accuracy over a rolling window.
    
    Usage:
        tracker = ModelAccuracyTracker()
        
        # On signal generation:
        tracker.record_prediction('trade_123', 'lstm', 'bullish')
        tracker.record_prediction('trade_123', 'lag-llama', 'bullish')
        
        # On trade close:
        tracker.record_outcome('trade_123', 'bullish')  # actual was bullish
        
        # Get weight for ensemble:
        w = tracker.get_weight('lstm')  # e.g. 1.3 if 75% accurate
    """
    
    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.window_size = self.config['window_size']
        
        # Per-model rolling accuracy: model_name -> deque of (predicted_correct: bool)
        self._model_results: Dict[str, deque] = {}
        
        # Pending predictions: trade_id -> {model_name: predicted_direction}
        self._pending: Dict[str, Dict[str, str]] = {}
        
        # Stats
        self._total_outcomes = 0
        
        # Load saved state
        self._load()
    
    def record_prediction(self, trade_id: str, model_name: str, direction: str):
        """Record a model's prediction for a trade.
        
        Args:
            trade_id: Unique trade identifier
            model_name: e.g. 'lstm', 'lag-llama', 'sentiment', 'dqn'
            direction: 'bullish' or 'bearish'
        """
        if not self.config.get('enabled', True):
            return
        
        if trade_id not in self._pending:
            self._pending[trade_id] = {}
        
        self._pending[trade_id][model_name] = direction.lower()
    
    def record_outcome(self, trade_id: str, actual_direction: str):
        """Record the actual trade outcome and update model accuracies.
        
        Args:
            trade_id: Trade that closed
            actual_direction: 'bullish' if profitable long or 'bearish' if profitable short
        """
        if not self.config.get('enabled', True):
            return
        
        predictions = self._pending.pop(trade_id, None)
        if not predictions:
            return
        
        actual = actual_direction.lower()
        
        for model_name, predicted_dir in predictions.items():
            correct = (predicted_dir == actual)
            
            if model_name not in self._model_results:
                self._model_results[model_name] = deque(maxlen=self.window_size)
            
            self._model_results[model_name].append(correct)
            
            accuracy = self._get_accuracy(model_name)
            logger.info(f"[ACCURACY] {model_name}: {'✓' if correct else '✗'} "
                       f"(predicted={predicted_dir}, actual={actual}, "
                       f"rolling={accuracy:.0%})")
        
        self._total_outcomes += 1
        
        # Auto-save periodically
        if self._total_outcomes % 10 == 0:
            self._save()
    
    def _get_accuracy(self, model_name: str) -> float:
        """Get rolling accuracy for a model (0.0 to 1.0)."""
        results = self._model_results.get(model_name)
        if not results or len(results) == 0:
            return 0.5  # Unknown = neutral
        return sum(results) / len(results)
    
    def get_weight(self, model_name: str) -> float:
        """Get dynamic weight multiplier for a model based on accuracy.
        
        Returns:
            Float between min_weight and max_weight.
            1.0 = neutral (50% accuracy or insufficient data)
            >1.0 = above-average accuracy, boost this model
            <1.0 = below-average accuracy, penalize this model
        """
        results = self._model_results.get(model_name)
        min_trades = self.config['min_trades']
        
        # Not enough data yet — use default
        if not results or len(results) < min_trades:
            return self.config['base_weight']
        
        accuracy = self._get_accuracy(model_name)
        
        # Linear scaling: 0% → min_weight, 50% → 1.0, 100% → max_weight
        base = self.config['base_weight']
        max_w = self.config['max_weight']
        min_w = self.config['min_weight']
        
        if accuracy >= 0.5:
            # Scale from 1.0 to max_weight as accuracy goes 50% → 100%
            weight = base + (max_w - base) * ((accuracy - 0.5) / 0.5)
        else:
            # Scale from min_weight to 1.0 as accuracy goes 0% → 50%
            weight = min_w + (base - min_w) * (accuracy / 0.5)
        
        return round(weight, 2)
    
    def get_all_weights(self) -> Dict[str, float]:
        """Get weights for all tracked models."""
        return {name: self.get_weight(name) for name in self._model_results}
    
    def get_stats(self) -> Dict:
        """Get accuracy stats for all models."""
        stats = {}
        for name in self._model_results:
            results = self._model_results[name]
            stats[name] = {
                'accuracy': round(self._get_accuracy(name), 3),
                'weight': self.get_weight(name),
                'trades_tracked': len(results),
                'correct': sum(results),
                'incorrect': len(results) - sum(results),
            }
        stats['total_outcomes'] = self._total_outcomes
        return stats
    
    def _save(self):
        """Save tracker state to disk."""
        save_path = self.config['save_path']
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data = {
                'total_outcomes': self._total_outcomes,
                'models': {}
            }
            for name, results in self._model_results.items():
                data['models'][name] = list(results)
            
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"[ACCURACY] Saved to {save_path}")
        except Exception as e:
            logger.warning(f"[ACCURACY] Save error: {e}")
    
    def _load(self):
        """Load tracker state from disk."""
        save_path = self.config['save_path']
        try:
            if os.path.exists(save_path):
                with open(save_path, 'r') as f:
                    data = json.load(f)
                
                self._total_outcomes = data.get('total_outcomes', 0)
                for name, results in data.get('models', {}).items():
                    self._model_results[name] = deque(results, maxlen=self.window_size)
                
                logger.info(f"[ACCURACY] Loaded {len(self._model_results)} model stats "
                           f"({self._total_outcomes} total outcomes)")
        except Exception as e:
            logger.debug(f"[ACCURACY] Load error: {e}")


# Singleton
_tracker = None

def get_accuracy_tracker(config: dict = None) -> ModelAccuracyTracker:
    """Get or create the singleton accuracy tracker."""
    global _tracker
    if _tracker is None:
        import config as app_config
        cfg = getattr(app_config, 'MODEL_ACCURACY_CONFIG', {})
        if config:
            cfg.update(config)
        _tracker = ModelAccuracyTracker(cfg)
    return _tracker
