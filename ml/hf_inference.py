"""
Hugging Face Inference API Client
Provides access to AI models via HF Inference API and Spaces.

Features:
- FinBERT sentiment analysis
- TimeGPT-style forecasting via Spaces
- Zero-shot classification
- Token-authenticated API access
"""
import os
import logging
import requests
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict
import json

logger = logging.getLogger(__name__)

# Load HF token from environment
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_API_BASE = "https://api-inference.huggingface.co/models"


@dataclass
class HFPrediction:
    """Container for HF model predictions."""
    model: str
    result: Dict
    confidence: float
    timestamp: datetime


class HuggingFaceClient:
    """
    Client for Hugging Face Inference API.
    
    Provides access to:
    - FinBERT: Financial sentiment analysis
    - Zero-shot classification for market regime
    - Text generation for trade reasoning
    """
    
    # Pre-configured models
    MODELS = {
        "finbert": "ProsusAI/finbert",
        "finbert_tone": "yiyanghkust/finbert-tone",
        "zero_shot": "facebook/bart-large-mnli",
        "summarizer": "facebook/bart-large-cnn",
    }
    
    def __init__(self, token: str = None):
        self.token = token or HF_TOKEN
        if not self.token:
            logger.warning("[HF] No Hugging Face token configured. Set HF_TOKEN in .env")
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        self._cache = {}
        self._cache_ttl = 500  # 5 minutes
        
        logger.info(f"[HF] Client initialized: token={'*****' + self.token[-4:] if self.token else 'MISSING'}")
    
    def _query(self, model_id: str, payload: Dict) -> Optional[Dict]:
        """Query HF Inference API."""
        if not self.token:
            logger.warning("[HF] Cannot query: no token")
            return None
        
        url = f"{HF_API_BASE}/{model_id}"
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # Model loading
                logger.info(f"[HF] Model {model_id} loading, please wait...")
                return None
            else:
                logger.warning(f"[HF] API error {response.status_code}: {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            logger.warning(f"[HF] Request timeout for {model_id}")
            return None
        except Exception as e:
            logger.error(f"[HF] Request error: {e}")
            return None
    
    def analyze_sentiment(self, texts: List[str], model: str = "finbert") -> Dict:
        """
        Analyze sentiment of financial texts using FinBERT.
        
        Args:
            texts: List of text strings to analyze
            model: 'finbert' or 'finbert_tone'
            
        Returns:
            Dict with sentiment scores and aggregated result
        """
        model_id = self.MODELS.get(model, self.MODELS["finbert"])
        
        results = []
        for text in texts[:10]:  # Limit to 10 texts
            # Check cache
            cache_key = f"sentiment:{model}:{hash(text)}"
            if cache_key in self._cache:
                cached_time, cached_result = self._cache[cache_key]
                if (datetime.now() - cached_time).seconds < self._cache_ttl:
                    results.append(cached_result)
                    continue
            
            response = self._query(model_id, {"inputs": text})
            
            if response:
                # Parse FinBERT response
                if isinstance(response, list) and len(response) > 0:
                    sentiment = response[0]  # First result
                    if isinstance(sentiment, list):
                        # Find highest score
                        best = max(sentiment, key=lambda x: x.get('score', 0))
                        result = {
                            'label': best.get('label', 'neutral'),
                            'score': best.get('score', 0),
                            'text': text[:50]
                        }
                    else:
                        result = sentiment
                else:
                    result = {'label': 'neutral', 'score': 0.5}
                
                results.append(result)
                self._cache[cache_key] = (datetime.now(), result)
        
        if not results:
            return {'direction': 'neutral', 'score': 0, 'confidence': 0}
        
        # Aggregate results
        positive_score = sum(r['score'] for r in results if r.get('label', '').lower() == 'positive')
        negative_score = sum(r['score'] for r in results if r.get('label', '').lower() == 'negative')
        
        total = len(results)
        avg_positive = positive_score / total
        avg_negative = negative_score / total
        
        if avg_positive > avg_negative + 0.1:
            direction = 'bullish'
            score = avg_positive
        elif avg_negative > avg_positive + 0.1:
            direction = 'bearish'
            score = -avg_negative
        else:
            direction = 'neutral'
            score = 0
        
        logger.info(f"[HF] Sentiment: {direction} (pos={avg_positive:.2f}, neg={avg_negative:.2f})")
        
        return {
            'direction': direction,
            'score': score,
            'confidence': max(avg_positive, avg_negative) * 100,
            'details': results
        }
    
    def classify_market_regime(self, description: str) -> Dict:
        """
        Classify market regime using zero-shot classification.
        
        Args:
            description: Text describing current market conditions
            
        Returns:
            Dict with regime classification
        """
        labels = ["trending bullish", "trending bearish", "ranging sideways", "high volatility"]
        
        response = self._query(self.MODELS["zero_shot"], {
            "inputs": description,
            "parameters": {"candidate_labels": labels}
        })
        
        if response and 'labels' in response and 'scores' in response:
            best_idx = 0
            best_score = response['scores'][0]
            
            for i, score in enumerate(response['scores']):
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            regime = response['labels'][best_idx]
            
            logger.info(f"[HF] Market regime: {regime} ({best_score:.1%})")
            
            return {
                'regime': regime,
                'confidence': best_score * 100,
                'all_scores': dict(zip(response['labels'], response['scores']))
            }
        
        return {'regime': 'unknown', 'confidence': 0}
    
    def get_news_summary(self, news_texts: List[str]) -> str:
        """
        Summarize multiple news articles into key points.
        
        Args:
            news_texts: List of news article texts
            
        Returns:
            Summarized text
        """
        combined = " ".join(news_texts[:5])[:2000]  # Limit text length
        
        response = self._query(self.MODELS["summarizer"], {
            "inputs": combined,
            "parameters": {"max_length": 150, "min_length": 30}
        })
        
        if response and isinstance(response, list) and len(response) > 0:
            summary = response[0].get('summary_text', '')
            logger.info(f"[HF] News summary: {summary[:100]}...")
            return summary
        
        return ""
    
    def get_trading_signal(self, symbol: str, news_texts: List[str] = None) -> Dict:
        """
        Get combined trading signal from HF models.
        
        Args:
            symbol: Trading symbol
            news_texts: Optional news headlines
            
        Returns:
            Dict with direction, confidence, and reasoning
        """
        result = {
            'symbol': symbol,
            'direction': 'neutral',
            'confidence': 0,
            'sentiment': None,
            'regime': None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Analyze sentiment if news provided
        if news_texts:
            sentiment = self.analyze_sentiment(news_texts)
            result['sentiment'] = sentiment
            result['direction'] = sentiment['direction']
            result['confidence'] = sentiment['confidence']
        
        return result


# Singleton instance
_hf_client = None


def get_hf_client() -> HuggingFaceClient:
    """Get or create the HF client singleton."""
    global _hf_client
    if _hf_client is None:
        _hf_client = HuggingFaceClient()
    return _hf_client


# Export for direct import
hf_client = None


def initialize_hf_client():
    """Initialize HF client (call at startup)."""
    global hf_client
    hf_client = get_hf_client()
    return hf_client
