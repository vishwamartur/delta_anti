"""
FastAPI Server for Delta Trading System
Production-ready REST API with ML predictions and WebSocket streaming
"""
import asyncio
from datetime import datetime
from typing import List, Optional, Dict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import trading system modules
from config import config
from data.market_data import market_data
from analysis.indicators import indicators
from analysis.signals import signal_generator, SignalType
from strategy.trade_manager import trade_manager

# Import ML modules
try:
    from ml.models.lstm_predictor import lstm_predictor
    from ml.sentiment.market_sentiment import sentiment_analyzer
    from ml.agents.dqn_trader import dqn_agent
    ML_AVAILABLE = True
except ImportError as e:
    print(f"[API] ML modules not available: {e}")
    ML_AVAILABLE = False

# Import Trade API
try:
    from api.trade_api import router as trade_router
    from strategy.advanced_trade_manager import initialize_trade_manager, get_trade_manager
    import config as cfg
    
    # Initialize trade manager
    initialize_trade_manager(
        account_balance=cfg.INITIAL_ACCOUNT_BALANCE,
        trade_config=cfg.TRADE_MANAGER_CONFIG
    )
    TRADE_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"[API] Trade manager not available: {e}")
    TRADE_MANAGER_AVAILABLE = False


# ============ Pydantic Models ============

class PredictionRequest(BaseModel):
    """Request model for price predictions."""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSD)")
    timeframe: str = Field(default="5m", description="Candle timeframe")
    lookback: int = Field(default=100, description="Number of candles to analyze")


class PredictionResponse(BaseModel):
    """Response model for price predictions."""
    symbol: str
    current_price: float
    predicted_prices: List[float]
    direction: str  # UP, DOWN, NEUTRAL
    confidence: float
    predicted_change_pct: float
    timestamp: str


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    symbol: str
    sentiment_score: float  # -100 to +100
    direction: str  # BULLISH, BEARISH, NEUTRAL
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    news_count: int
    top_headlines: List[str]


class SignalResponse(BaseModel):
    """Response model for trading signals."""
    symbol: str
    signal_type: str  # LONG, SHORT, NEUTRAL, EXIT
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    reasons: List[str]
    ml_prediction: Optional[Dict] = None
    sentiment: Optional[Dict] = None


class TradeRequest(BaseModel):
    """Request model for trade execution."""
    symbol: str
    direction: str  # LONG, SHORT
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class TradeResponse(BaseModel):
    """Response model for trade results."""
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    size: float
    status: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    ml_available: bool
    active_connections: int
    uptime_seconds: float


# ============ FastAPI App ============

app = FastAPI(
    title="Delta Anti Trading API",
    description="ML-powered crypto trading system API with real-time predictions",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include trade API router
if TRADE_MANAGER_AVAILABLE:
    app.include_router(trade_router)

# Track startup time and connections
startup_time = datetime.now()
active_websockets: List[WebSocket] = []


# ============ REST Endpoints ============

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        ml_available=ML_AVAILABLE,
        active_connections=len(active_websockets),
        uptime_seconds=(datetime.now() - startup_time).total_seconds()
    )


@app.get("/api/v2/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        ml_available=ML_AVAILABLE,
        active_connections=len(active_websockets),
        uptime_seconds=(datetime.now() - startup_time).total_seconds()
    )


@app.post("/api/v2/predict", response_model=PredictionResponse)
async def get_price_prediction(request: PredictionRequest):
    """
    Get ML price predictions for a symbol.
    
    Uses LSTM model to predict next 5 price points.
    """
    if not ML_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="ML models not available"
        )
    
    try:
        # Load historical data if not present
        if not market_data.has_data(request.symbol):
            success = market_data.load_historical_candles(
                request.symbol, request.timeframe, 24
            )
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data available for {request.symbol}"
                )
        
        # Get DataFrame
        df = market_data.get_dataframe(request.symbol)
        if df is None or len(df) < 50:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for prediction"
            )
        
        # Make prediction
        prediction = lstm_predictor.predict(df)
        
        return PredictionResponse(
            symbol=request.symbol,
            current_price=prediction.current_price,
            predicted_prices=prediction.predicted_prices,
            direction=prediction.direction,
            confidence=prediction.confidence,
            predicted_change_pct=prediction.predicted_change_pct,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/sentiment/{symbol}", response_model=SentimentResponse)
async def get_sentiment(symbol: str):
    """
    Get market sentiment analysis for a symbol.
    
    Analyzes recent news using FinBERT.
    """
    if not ML_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ML models not available"
        )
    
    try:
        sentiment = sentiment_analyzer.analyze_symbol(symbol)
        signal = sentiment_analyzer.get_sentiment_signal(symbol)
        
        return SentimentResponse(
            symbol=symbol,
            sentiment_score=sentiment.score,
            direction=signal['direction'],
            positive_pct=sentiment.positive_pct,
            negative_pct=sentiment.negative_pct,
            neutral_pct=sentiment.neutral_pct,
            news_count=sentiment.news_count,
            top_headlines=sentiment.top_headlines
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/signals/{symbol}", response_model=SignalResponse)
async def get_trading_signal(
    symbol: str,
    timeframe: str = Query(default="5m", description="Candle timeframe"),
    include_ml: bool = Query(default=True, description="Include ML predictions")
):
    """
    Get combined trading signals for a symbol.
    
    Combines technical analysis with ML predictions and sentiment.
    """
    try:
        # Load data
        if not market_data.has_data(symbol):
            market_data.load_historical_candles(symbol, timeframe, 24)
        
        df = market_data.get_dataframe(symbol)
        if df is None or len(df) < 50:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for analysis"
            )
        
        # Calculate indicators
        ind = indicators.calculate_all(df)
        if ind is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to calculate indicators"
            )
        
        # Generate technical signal
        signal = signal_generator.generate_signal(symbol, ind)
        
        # Add ML components if available
        ml_prediction = None
        sentiment = None
        
        if ML_AVAILABLE and include_ml:
            try:
                pred = lstm_predictor.predict(df)
                ml_prediction = {
                    'direction': pred.direction,
                    'confidence': pred.confidence,
                    'predicted_change_pct': pred.predicted_change_pct
                }
            except:
                pass
            
            try:
                sent = sentiment_analyzer.get_sentiment_signal(symbol)
                sentiment = {
                    'direction': sent['direction'],
                    'score': sent['score'],
                    'strength': sent['strength']
                }
            except:
                pass
        
        return SignalResponse(
            symbol=symbol,
            signal_type=signal.signal_type.value,
            confidence=signal.confidence,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_reward_ratio=signal.risk_reward_ratio,
            reasons=signal.reasons,
            ml_prediction=ml_prediction,
            sentiment=sentiment
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/indicators/{symbol}")
async def get_indicators(
    symbol: str,
    timeframe: str = Query(default="5m")
):
    """Get current technical indicator values."""
    try:
        if not market_data.has_data(symbol):
            market_data.load_historical_candles(symbol, timeframe, 24)
        
        df = market_data.get_dataframe(symbol)
        if df is None:
            raise HTTPException(status_code=404, detail="No data")
        
        ind = indicators.calculate_all(df)
        if ind is None:
            raise HTTPException(status_code=400, detail="Calculation failed")
        
        return {
            'symbol': symbol,
            'price': ind.price,
            'rsi': ind.rsi,
            'macd': ind.macd,
            'macd_signal': ind.macd_signal,
            'macd_histogram': ind.macd_histogram,
            'bb_upper': ind.bb_upper,
            'bb_middle': ind.bb_middle,
            'bb_lower': ind.bb_lower,
            'bb_percent': ind.bb_percent,
            'atr': ind.atr,
            'adx': ind.adx,
            'ema_fast': ind.ema_fast,
            'ema_slow': ind.ema_slow,
            'trend_strength': ind.trend_strength,
            'timestamp': datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/trades")
async def get_open_trades():
    """Get all open trades."""
    trades = trade_manager.get_open_trades()
    return {
        'count': len(trades),
        'trades': [
            {
                'trade_id': t.trade_id,
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'current_price': t.current_price,
                'size': t.size,
                'pnl': t.unrealized_pnl,
                'pnl_percent': t.pnl_percent,
                'stop_loss': t.stop_loss,
                'take_profit': t.take_profit
            }
            for t in trades
        ]
    }


@app.get("/api/v2/stats")
async def get_trading_stats():
    """Get trading statistics."""
    stats = trade_manager.get_stats()
    return stats


# ============ WebSocket Endpoints ============

@app.websocket("/ws/predictions/{symbol}")
async def websocket_predictions(websocket: WebSocket, symbol: str):
    """
    Stream real-time predictions via WebSocket.
    
    Sends updates every 5 seconds.
    """
    await websocket.accept()
    active_websockets.append(websocket)
    
    try:
        while True:
            # Load latest data
            if not market_data.has_data(symbol):
                market_data.load_historical_candles(symbol, "5m", 4)
            
            df = market_data.get_dataframe(symbol)
            
            data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add indicators
            if df is not None and len(df) >= 50:
                ind = indicators.calculate_all(df)
                if ind:
                    data['price'] = ind.price
                    data['rsi'] = ind.rsi
                    data['macd_histogram'] = ind.macd_histogram
                    
                    # Generate signal
                    signal = signal_generator.generate_signal(symbol, ind)
                    data['signal'] = signal.signal_type.value
                    data['confidence'] = signal.confidence
                
                # Add ML prediction
                if ML_AVAILABLE:
                    try:
                        pred = lstm_predictor.predict(df)
                        data['ml_direction'] = pred.direction
                        data['ml_confidence'] = pred.confidence
                    except:
                        pass
            
            await websocket.send_json(data)
            await asyncio.sleep(5)
    
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)


@app.websocket("/ws/signals")
async def websocket_all_signals(websocket: WebSocket):
    """
    Stream signals for all configured symbols.
    """
    await websocket.accept()
    active_websockets.append(websocket)
    
    symbols = config.TRADING_SYMBOLS
    
    try:
        while True:
            signals = []
            
            for symbol in symbols:
                try:
                    if not market_data.has_data(symbol):
                        market_data.load_historical_candles(symbol, "5m", 4)
                    
                    df = market_data.get_dataframe(symbol)
                    if df is not None and len(df) >= 50:
                        ind = indicators.calculate_all(df)
                        if ind:
                            signal = signal_generator.generate_signal(symbol, ind)
                            signals.append({
                                'symbol': symbol,
                                'type': signal.signal_type.value,
                                'confidence': signal.confidence,
                                'price': ind.price
                            })
                except:
                    continue
            
            await websocket.send_json({
                'timestamp': datetime.now().isoformat(),
                'signals': signals
            })
            
            await asyncio.sleep(10)
    
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)


# ============ Run Server ============

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    print(f"[API] Starting server at http://{host}:{port}")
    print(f"[API] Documentation at http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
