"""
Market Data Manager - Handles real-time and historical OHLC data
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
import pandas as pd
import numpy as np
import config
from api.delta_rest import rest_client


@dataclass
class Candle:
    """Single OHLC candle data."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


@dataclass
class TickerData:
    """Real-time ticker data."""
    symbol: str
    last_price: float
    mark_price: float
    best_bid: float
    best_ask: float
    bid_size: float
    ask_size: float
    volume_24h: float
    price_change_24h: float
    open_interest: float
    timestamp: int
    
    # Greeks for options
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_volatility: Optional[float] = None


class MarketDataManager:
    """
    Manages market data for multiple symbols.
    Stores candles and provides data for technical analysis.
    """
    
    def __init__(self, max_candles: int = None):
        self.max_candles = max_candles or config.CANDLE_HISTORY_SIZE
        
        # Candle storage per symbol {symbol: deque of Candles}
        self._candles: Dict[str, Deque[Candle]] = {}
        
        # Latest ticker data per symbol
        self._tickers: Dict[str, TickerData] = {}
        
        # DataFrame cache for analysis (updated on demand)
        self._df_cache: Dict[str, pd.DataFrame] = {}
        self._cache_valid: Dict[str, bool] = {}
    
    def _get_candle_deque(self, symbol: str) -> Deque[Candle]:
        """Get or create candle deque for symbol."""
        if symbol not in self._candles:
            self._candles[symbol] = deque(maxlen=self.max_candles)
            self._cache_valid[symbol] = False
        return self._candles[symbol]
    
    def add_candle(self, symbol: str, candle: Candle):
        """Add a new candle for a symbol."""
        candles = self._get_candle_deque(symbol)
        
        # Check if this replaces the current candle or adds new one
        if candles and candles[-1].timestamp == candle.timestamp:
            # Update existing candle (ongoing candle)
            candles[-1] = candle
        else:
            # Add new candle
            candles.append(candle)
        
        # Invalidate cache
        self._cache_valid[symbol] = False
    
    def update_from_ws_candlestick(self, data: Dict):
        """Update candles from WebSocket candlestick message."""
        symbol = data.get('symbol', '')
        if not symbol or symbol.startswith('MARK:'):
            return
            
        candle = Candle(
            timestamp=data.get('candle_start_time', 0),
            open=float(data.get('open', 0)),
            high=float(data.get('high', 0)),
            low=float(data.get('low', 0)),
            close=float(data.get('close', 0)),
            volume=float(data.get('volume', 0))
        )
        self.add_candle(symbol, candle)
    
    def update_ticker(self, data: Dict):
        """Update ticker from WebSocket ticker message."""
        symbol = data.get('symbol', '')
        if not symbol:
            return
        
        quotes = data.get('quotes', {})
        greeks = data.get('greeks', {})
        
        ticker = TickerData(
            symbol=symbol,
            last_price=float(data.get('close', 0)),
            mark_price=float(data.get('mark_price', 0)),
            best_bid=float(quotes.get('best_bid', 0)),
            best_ask=float(quotes.get('best_ask', 0)),
            bid_size=float(quotes.get('bid_size', 0)),
            ask_size=float(quotes.get('ask_size', 0)),
            volume_24h=float(data.get('volume', 0)),
            price_change_24h=float(data.get('mark_change_24h', 0)),
            open_interest=float(data.get('oi', 0)),
            timestamp=data.get('timestamp', 0),
            delta=float(greeks.get('delta', 0)) if greeks else None,
            gamma=float(greeks.get('gamma', 0)) if greeks else None,
            theta=float(greeks.get('theta', 0)) if greeks else None,
            vega=float(greeks.get('vega', 0)) if greeks else None,
            implied_volatility=float(quotes.get('mark_iv', 0)) if quotes.get('mark_iv') else None
        )
        self._tickers[symbol] = ticker
    
    def load_historical_candles(self, symbol: str, resolution: str = None, 
                                 lookback_hours: int = 24):
        """Load historical candles from REST API."""
        resolution = resolution or config.DEFAULT_TIMEFRAME
        
        end_time = int(time.time())
        start_time = end_time - (lookback_hours * 3600)
        
        response = rest_client.get_candles(
            symbol=symbol,
            resolution=resolution,
            start=start_time,
            end=end_time
        )
        
        if 'result' in response:
            candles_data = response['result']
            candles = self._get_candle_deque(symbol)
            candles.clear()
            
            for c in candles_data:
                candle = Candle(
                    timestamp=c.get('time', 0),
                    open=float(c.get('open', 0)),
                    high=float(c.get('high', 0)),
                    low=float(c.get('low', 0)),
                    close=float(c.get('close', 0)),
                    volume=float(c.get('volume', 0))
                )
                candles.append(candle)
            
            self._cache_valid[symbol] = False
            print(f"[DATA] Loaded {len(candles)} candles for {symbol}")
            return True
        else:
            print(f"[DATA] Failed to load candles for {symbol}: {response}")
            return False
    
    def get_dataframe(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get candle data as pandas DataFrame for analysis.
        Returns None if no data available.
        """
        if symbol not in self._candles or not self._candles[symbol]:
            return None
        
        # Use cached DataFrame if valid
        if self._cache_valid.get(symbol) and symbol in self._df_cache:
            return self._df_cache[symbol]
        
        # Build DataFrame from candles
        candles = self._candles[symbol]
        data = [c.to_dict() for c in candles]
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')
        df.set_index('datetime', inplace=True)
        
        # Cache and return
        self._df_cache[symbol] = df
        self._cache_valid[symbol] = True
        return df
    
    def get_ticker(self, symbol: str) -> Optional[TickerData]:
        """Get latest ticker data for symbol."""
        return self._tickers.get(symbol)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol."""
        # Try ticker first
        ticker = self._tickers.get(symbol)
        if ticker:
            return ticker.last_price
        
        # Fall back to last candle close
        if symbol in self._candles and self._candles[symbol]:
            return self._candles[symbol][-1].close
        
        return None
    
    def get_candle_count(self, symbol: str) -> int:
        """Get number of candles stored for symbol."""
        return len(self._candles.get(symbol, []))
    
    @property
    def symbols(self) -> List[str]:
        """Get list of all symbols with data."""
        return list(set(list(self._candles.keys()) + list(self._tickers.keys())))


# Singleton instance
market_data = MarketDataManager()
