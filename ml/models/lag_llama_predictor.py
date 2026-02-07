"""
Lag-Llama Time Series Foundation Model for Price Forecasting
A probabilistic forecasting model pre-trained on 100+ datasets.

Optimized for GTX 1650 Ti (4GB VRAM) - inference only, no fine-tuning.
For GPU training, 12GB+ VRAM recommended.
"""
import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

import sys
logger = logging.getLogger(__name__)

# Add vendor directory to path for lag-llama
VENDOR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vendor", "lag-llama")
if os.path.exists(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.append(VENDOR_DIR)
    logger.info(f"[LAG-LLAMA] Added vendor path: {VENDOR_DIR}")

# Check for required dependencies
try:
    import torch
    TORCH_AVAILABLE = True
    
    # Check for CUDA
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"[LAG-LLAMA] GPU detected: {GPU_NAME} ({GPU_MEMORY:.1f}GB)")
    else:
        logger.info("[LAG-LLAMA] No GPU detected, using CPU")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    logger.warning("[LAG-LLAMA] PyTorch not installed")

try:
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    GLUONTS_AVAILABLE = True
except ImportError:
    GLUONTS_AVAILABLE = False
    logger.warning("[LAG-LLAMA] GluonTS not installed. Run: pip install gluonts")

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("[LAG-LLAMA] huggingface_hub not installed. Run: pip install huggingface-hub")


@dataclass
class LagLlamaForecast:
    """Container for Lag-Llama forecast results."""
    symbol: str
    current_price: float
    predicted_prices: List[float]
    predicted_low: List[float]   # Lower quantile (10%)
    predicted_high: List[float]  # Upper quantile (90%)
    direction: str               # 'bullish', 'bearish', or 'neutral'
    confidence: float            # 0-1 confidence score
    predicted_change_pct: float  # Expected % change
    forecast_horizon: int        # Number of steps forecasted
    timestamp: datetime


class LagLlamaPredictor:
    """
    Lag-Llama Time Series Foundation Model for trading predictions.
    
    Features:
    - Zero-shot forecasting (no training needed)
    - Probabilistic predictions with confidence intervals
    - Optimized for 4GB GPU (GTX 1650 Ti)
    - Falls back to CPU if GPU unavailable
    """
    
    MODEL_REPO = "time-series-foundation-models/Lag-Llama"
    MODEL_FILENAME = "lag-llama.ckpt"
    
    def __init__(
        self,
        prediction_length: int = 24,    # Forecast 24 steps ahead
        context_length: int = 256,      # Look-back window
        num_samples: int = 50,          # Probabilistic samples (reduced for 4GB GPU)
        device: str = "auto",           # 'cuda', 'cpu', or 'auto'
        batch_size: int = 1             # Small batch for low VRAM
    ):
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.num_samples = num_samples
        self.batch_size = batch_size
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.is_initialized = False
        self.model_path = None
        
        # Cache directory for model
        self.cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "lag_llama")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"[LAG-LLAMA] Initialized: device={self.device}, "
                   f"context={context_length}, horizon={prediction_length}")
    
    def _download_model(self) -> str:
        """Download Lag-Llama model from Hugging Face."""
        if not HF_HUB_AVAILABLE:
            raise RuntimeError("huggingface_hub not installed")
        
        local_path = os.path.join(self.cache_dir, self.MODEL_FILENAME)
        
        if os.path.exists(local_path):
            logger.info(f"[LAG-LLAMA] Model found at {local_path}")
            return local_path
        
        logger.info("[LAG-LLAMA] Downloading model from Hugging Face...")
        try:
            model_path = hf_hub_download(
                repo_id=self.MODEL_REPO,
                filename=self.MODEL_FILENAME,
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False
            )
            logger.info(f"[LAG-LLAMA] Model downloaded to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"[LAG-LLAMA] Failed to download model: {e}")
            raise
    
    def _initialize_model(self):
        """Initialize the Lag-Llama model."""
        if self.is_initialized:
            return
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed")
        
        try:
            # Download model if needed
            self.model_path = self._download_model()
            
            # Load checkpoint
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract model configuration
            model_kwargs = ckpt.get("hyper_parameters", {}).get("model_kwargs", {})
            
            # Import Lag-Llama estimator (lazy import)
            from lag_llama.gluon.estimator import LagLlamaEstimator
            
            # Create estimator with reduced settings for 4GB GPU
            self.estimator = LagLlamaEstimator(
                ckpt_path=self.model_path,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                num_samples=self.num_samples,
                device=self.device,
                batch_size=self.batch_size,
                nonnegative_pred_samples=True,  # Prices are positive
                **model_kwargs
            )
            
            # Create predictor
            self.model = self.estimator.create_predictor(
                transformation=self.estimator.create_transformation()
            )
            
            self.is_initialized = True
            logger.info("[LAG-LLAMA] Model initialized successfully")
            
        except ImportError:
            logger.error("[LAG-LLAMA] lag_llama package not installed. "
                        "Clone: git clone https://github.com/time-series-foundation-models/lag-llama")
            raise
        except Exception as e:
            logger.error(f"[LAG-LLAMA] Model initialization failed: {e}")
            raise
    
    def _prepare_data(self, df: pd.DataFrame, symbol: str = "BTCUSD") -> 'PandasDataset':
        """Prepare DataFrame for Lag-Llama input."""
        if not GLUONTS_AVAILABLE:
            raise RuntimeError("GluonTS not installed")
        
        # Ensure proper column names
        if 'close' in df.columns:
            target_col = 'close'
        elif 'Close' in df.columns:
            target_col = 'Close'
        else:
            raise ValueError("DataFrame must have 'close' or 'Close' column")
        
        # Ensure datetime index
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        elif 'date' in df.columns:
            df = df.set_index('date')
        elif 'Date' in df.columns:
            df = df.set_index('Date')
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Create GluonTS dataset
        dataset = PandasDataset.from_long_dataframe(
            df.reset_index(),
            item_id=symbol,
            timestamp="index" if "index" in df.reset_index().columns else df.reset_index().columns[0],
            target=target_col
        )
        
        return dataset
    
    def predict(self, df: pd.DataFrame, symbol: str = "BTCUSD") -> Optional[LagLlamaForecast]:
        """
        Generate price forecast from historical data.
        
        Args:
            df: DataFrame with OHLCV data (must have 'close' column)
            symbol: Trading symbol for identification
            
        Returns:
            LagLlamaForecast with predictions and confidence intervals
        """
        try:
            # Initialize model on first use
            if not self.is_initialized:
                self._initialize_model()
            
            # Prepare data
            dataset = self._prepare_data(df, symbol)
            
            # Get current price
            current_price = float(df['close'].iloc[-1] if 'close' in df.columns 
                                 else df['Close'].iloc[-1])
            
            # Generate forecasts
            forecasts = list(self.model.predict(dataset))
            
            if not forecasts:
                logger.warning("[LAG-LLAMA] No forecasts generated")
                return None
            
            forecast = forecasts[0]
            
            # Extract statistics from probabilistic forecast
            predicted_prices = forecast.mean.tolist()
            predicted_low = forecast.quantile(0.1).tolist()   # 10th percentile
            predicted_high = forecast.quantile(0.9).tolist()  # 90th percentile
            
            # Calculate direction and confidence
            final_price = predicted_prices[-1]
            price_change = final_price - current_price
            change_pct = (price_change / current_price) * 100
            
            # Determine direction
            if change_pct > 0.5:
                direction = "bullish"
            elif change_pct < -0.5:
                direction = "bearish"
            else:
                direction = "neutral"
            
            # Calculate confidence based on prediction interval width
            interval_widths = [(h - l) / current_price for h, l in zip(predicted_high, predicted_low)]
            avg_interval = sum(interval_widths) / len(interval_widths)
            # Narrower intervals = higher confidence
            confidence = max(0.1, min(0.95, 1.0 - avg_interval * 10))
            
            result = LagLlamaForecast(
                symbol=symbol,
                current_price=current_price,
                predicted_prices=predicted_prices,
                predicted_low=predicted_low,
                predicted_high=predicted_high,
                direction=direction,
                confidence=confidence,
                predicted_change_pct=change_pct,
                forecast_horizon=self.prediction_length,
                timestamp=datetime.now()
            )
            
            logger.info(f"[LAG-LLAMA] {symbol}: {direction} "
                       f"({change_pct:+.2f}%, conf={confidence:.1%})")
            
            return result
            
        except Exception as e:
            logger.error(f"[LAG-LLAMA] Prediction error: {e}")
            return self._fallback_prediction(df, symbol)
    
    def _fallback_prediction(self, df: pd.DataFrame, symbol: str) -> LagLlamaForecast:
        """Simple momentum-based fallback when Lag-Llama unavailable."""
        logger.info("[LAG-LLAMA] Using fallback momentum predictor")
        
        close_col = 'close' if 'close' in df.columns else 'Close'
        prices = df[close_col].values
        current_price = float(prices[-1])
        
        # Simple momentum: average of recent returns
        returns = np.diff(prices[-21:]) / prices[-21:-1]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Project forward
        predicted_prices = []
        predicted_low = []
        predicted_high = []
        price = current_price
        
        for i in range(self.prediction_length):
            price = price * (1 + avg_return)
            predicted_prices.append(price)
            predicted_low.append(price * (1 - std_return * 1.5))
            predicted_high.append(price * (1 + std_return * 1.5))
        
        change_pct = ((predicted_prices[-1] - current_price) / current_price) * 100
        
        if change_pct > 0.5:
            direction = "bullish"
        elif change_pct < -0.5:
            direction = "bearish"
        else:
            direction = "neutral"
        
        return LagLlamaForecast(
            symbol=symbol,
            current_price=current_price,
            predicted_prices=predicted_prices,
            predicted_low=predicted_low,
            predicted_high=predicted_high,
            direction=direction,
            confidence=0.5,  # Lower confidence for fallback
            predicted_change_pct=change_pct,
            forecast_horizon=self.prediction_length,
            timestamp=datetime.now()
        )
    
    def get_trading_signal(self, df: pd.DataFrame, symbol: str = "BTCUSD") -> Dict:
        """
        Get a trading signal based on Lag-Llama forecast.
        
        Returns:
            Dict with 'direction', 'confidence', 'change_pct', 'action'
        """
        forecast = self.predict(df, symbol)
        
        if forecast is None:
            return {"direction": "neutral", "confidence": 0, "change_pct": 0, "action": "HOLD"}
        
        # Determine action based on direction and confidence
        if forecast.direction == "bullish" and forecast.confidence > 0.6:
            action = "BUY"
        elif forecast.direction == "bearish" and forecast.confidence > 0.6:
            action = "SELL"
        else:
            action = "HOLD"
        
        return {
            "direction": forecast.direction,
            "confidence": forecast.confidence,
            "change_pct": forecast.predicted_change_pct,
            "action": action,
            "predicted_price": forecast.predicted_prices[-1] if forecast.predicted_prices else None,
            "horizon": forecast.forecast_horizon
        }


# Singleton instance (lazy initialization)
_lag_llama_predictor = None

def get_lag_llama_predictor() -> LagLlamaPredictor:
    """Get or create the Lag-Llama predictor singleton."""
    global _lag_llama_predictor
    if _lag_llama_predictor is None:
        _lag_llama_predictor = LagLlamaPredictor()
    return _lag_llama_predictor


# Convenience export
lag_llama_predictor = None  # Will be initialized on first use


def initialize_lag_llama():
    """Initialize Lag-Llama predictor (call this to pre-load model)."""
    global lag_llama_predictor
    lag_llama_predictor = get_lag_llama_predictor()
    return lag_llama_predictor
