"""
Market Sentiment Analysis
Analyzes news and social media sentiment for trading signals.
Now with Hugging Face Inference API fallback.
"""
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)

# Try importing transformers for local FinBERT
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.info("[ML] Transformers not installed locally, will use HF Inference API")

# Try importing HF Inference API client
try:
    from ml.hf_inference import get_hf_client
    HF_API_AVAILABLE = True
except ImportError:
    HF_API_AVAILABLE = False


@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""
    symbol: str
    score: float  # -100 to +100
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    news_count: int
    timestamp: datetime
    top_headlines: List[str]


class SentimentAnalyzer:
    """
    Analyzes market sentiment from news and social media.
    Uses FinBERT for financial text classification.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._initialize_model()
        
        # Cache for sentiment results
        self._cache: Dict[str, SentimentResult] = {}
        self._cache_duration = timedelta(minutes=15)
    
    def _initialize_model(self):
        """Initialize FinBERT model if available."""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Suppress warnings about chat templates (not applicable to FinBERT)
                import warnings
                import logging
                logging.getLogger("transformers").setLevel(logging.ERROR)
                warnings.filterwarnings("ignore", message=".*chat_templates.*")
                
                self.model = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    device=-1  # Use CPU to avoid GPU memory issues
                )
                
                # Restore logging level
                logging.getLogger("transformers").setLevel(logging.WARNING)
                
                print("[ML] FinBERT sentiment model loaded")
            except Exception as e:
                print(f"[ML] Failed to load FinBERT: {e}")
                self.model = None
    
    def get_crypto_news(self, symbol: str = "BTC", limit: int = 50) -> List[Dict]:
        """
        Fetch recent crypto news from CryptoCompare API.
        
        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)
            limit: Maximum number of articles
            
        Returns:
            List of news articles
        """
        # Extract base symbol from trading pair
        base_symbol = symbol.replace("USD", "").replace("USDT", "")
        
        url = "https://min-api.cryptocompare.com/data/v2/news/"
        params = {
            "lang": "EN",
            "categories": base_symbol,
            "excludeCategories": "Sponsored"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("Response") == "Success":
                articles = data.get("Data", [])[:limit]
                return articles
        except Exception as e:
            print(f"[ML] Error fetching news: {e}")
        
        return []
    
    def analyze_text(self, texts: List[str]) -> Dict:
        """
        Analyze sentiment of text array.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Dict with sentiment scores and percentages
        """
        if not texts:
            return {
                'score': 0,
                'positive_pct': 0,
                'negative_pct': 0,
                'neutral_pct': 1.0
            }
        
        if self.model:
            # Use FinBERT
            try:
                # Truncate texts to model max length
                truncated_texts = [t[:512] for t in texts]
                results = self.model(truncated_texts)
                
                positive = sum(1 for r in results if r['label'] == 'positive')
                negative = sum(1 for r in results if r['label'] == 'negative')
                neutral = sum(1 for r in results if r['label'] == 'neutral')
                
                total = len(results)
                sentiment_score = ((positive - negative) / total) * 100
                
                return {
                    'score': sentiment_score,
                    'positive_pct': positive / total,
                    'negative_pct': negative / total,
                    'neutral_pct': neutral / total
                }
            except Exception as e:
                logger.warning(f"[ML] FinBERT error: {e}")
        
        # Fallback 1: Use HF Inference API if available
        if HF_API_AVAILABLE:
            try:
                hf_client = get_hf_client()
                hf_result = hf_client.analyze_sentiment(texts)
                if hf_result and hf_result.get('confidence', 0) > 0:
                    logger.info(f"[HF API] Sentiment: {hf_result['direction']}")
                    # Convert HF format to our format
                    score = hf_result.get('score', 0) * 100
                    return {
                        'score': score,
                        'positive_pct': 0.5 + score/200 if score > 0 else 0.3,
                        'negative_pct': 0.5 - score/200 if score < 0 else 0.3,
                        'neutral_pct': 0.4
                    }
            except Exception as e:
                logger.debug(f"[HF API] Fallback error: {e}")
        
        # Fallback 2: Simple keyword-based sentiment
        return self._simple_sentiment(texts)
    
    def _simple_sentiment(self, texts: List[str]) -> Dict:
        """Simple keyword-based sentiment analysis fallback."""
        positive_words = [
            'bullish', 'surge', 'rally', 'gain', 'up', 'high', 'growth',
            'positive', 'profit', 'buy', 'strong', 'boom', 'soar', 'jump',
            'breakthrough', 'adoption', 'partnership', 'innovation'
        ]
        
        negative_words = [
            'bearish', 'crash', 'drop', 'fall', 'down', 'low', 'decline',
            'negative', 'loss', 'sell', 'weak', 'bust', 'plunge', 'dump',
            'hack', 'scam', 'fraud', 'ban', 'regulation', 'lawsuit'
        ]
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for text in texts:
            text_lower = text.lower()
            
            pos_matches = sum(1 for w in positive_words if w in text_lower)
            neg_matches = sum(1 for w in negative_words if w in text_lower)
            
            if pos_matches > neg_matches:
                positive_count += 1
            elif neg_matches > pos_matches:
                negative_count += 1
            else:
                neutral_count += 1
        
        total = len(texts)
        sentiment_score = ((positive_count - negative_count) / total) * 100
        
        return {
            'score': sentiment_score,
            'positive_pct': positive_count / total,
            'negative_pct': negative_count / total,
            'neutral_pct': neutral_count / total
        }
    
    def analyze_symbol(self, symbol: str, use_cache: bool = True) -> SentimentResult:
        """
        Get complete sentiment analysis for a trading symbol.
        
        Args:
            symbol: Trading symbol (e.g., BTCUSD)
            use_cache: Whether to use cached results
            
        Returns:
            SentimentResult with all sentiment data
        """
        # Check cache
        if use_cache and symbol in self._cache:
            cached = self._cache[symbol]
            if datetime.now() - cached.timestamp < self._cache_duration:
                return cached
        
        # Fetch news
        articles = self.get_crypto_news(symbol)
        
        if not articles:
            # Return neutral sentiment if no news
            result = SentimentResult(
                symbol=symbol,
                score=0,
                positive_pct=0,
                negative_pct=0,
                neutral_pct=1.0,
                news_count=0,
                timestamp=datetime.now(),
                top_headlines=[]
            )
            return result
        
        # Extract headlines and bodies
        texts = []
        headlines = []
        
        for article in articles:
            title = article.get('title', '')
            body = article.get('body', '')[:500]  # Limit body length
            
            if title:
                headlines.append(title)
                texts.append(f"{title}. {body}")
        
        # Analyze sentiment
        sentiment = self.analyze_text(texts)
        
        result = SentimentResult(
            symbol=symbol,
            score=sentiment['score'],
            positive_pct=sentiment['positive_pct'],
            negative_pct=sentiment['negative_pct'],
            neutral_pct=sentiment['neutral_pct'],
            news_count=len(articles),
            timestamp=datetime.now(),
            top_headlines=headlines[:5]
        )
        
        # Update cache
        self._cache[symbol] = result
        
        return result
    
    def get_sentiment_signal(self, symbol: str) -> Dict:
        """
        Get sentiment as a trading signal component.
        
        Returns:
            Dict with signal direction and strength
        """
        sentiment = self.analyze_symbol(symbol)
        
        # Determine signal
        if sentiment.score > 20:
            direction = "BULLISH"
            strength = min(100, sentiment.score)
        elif sentiment.score < -20:
            direction = "BEARISH"
            strength = min(100, abs(sentiment.score))
        else:
            direction = "NEUTRAL"
            strength = 50 - abs(sentiment.score)
        
        return {
            'direction': direction,
            'strength': strength,
            'score': sentiment.score,
            'news_count': sentiment.news_count,
            'headlines': sentiment.top_headlines
        }


# Singleton instance
sentiment_analyzer = SentimentAnalyzer()
