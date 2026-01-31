# Delta Exchange Real-Time Trading System

A Python-based quantitative trading system for Delta Exchange with **ML-powered predictions** and **REST API server**.

## ✨ Features

### Core Trading
- **Real-Time Data**: WebSocket streaming of OHLC candles and tickers
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX, EMA/SMA
- **Trade Signals**: Long/Short entry with confidence scores
- **Risk Management**: Dynamic TP/SL based on ATR

### 🤖 ML Layer (v2.0)
- **LSTM Price Predictor**: Bidirectional LSTM with attention mechanism
- **Sentiment Analysis**: FinBERT-powered news sentiment scoring
- **DQN Trading Agent**: Reinforcement learning for trade decisions
- **Feature Engineering**: 100+ features from OHLCV data

### 🌐 REST API (v2.0)
- **FastAPI Server**: Production-ready REST endpoints
- **WebSocket Streaming**: Real-time predictions
- **Webhooks**: TradingView & Telegram integration

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/vishwamartur/delta_anti.git
cd delta_anti

# Install dependencies
pip install -r requirements.txt

# For ML features (optional, requires more dependencies)
pip install torch transformers scikit-learn
```

## ⚙️ Configuration

1. Copy `.env.example` to `.env`
2. Add your Delta Exchange API credentials:
   ```
   DELTA_API_KEY=your_api_key
   DELTA_API_SECRET=your_api_secret
   ```

> ⚠️ Ensure your IP is whitelisted on Delta Exchange for API access.

## 🚀 Usage

### Console Dashboard
```bash
python main.py
```

### API Server
```bash
python -m api.server.main
```
Server runs at `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/predict` | POST | Get ML price predictions |
| `/api/v2/signals/{symbol}` | GET | Get combined trading signals |
| `/api/v2/sentiment/{symbol}` | GET | Get sentiment analysis |
| `/api/v2/indicators/{symbol}` | GET | Get technical indicators |
| `/api/v2/trades` | GET | Get open trades |
| `/ws/predictions/{symbol}` | WebSocket | Stream predictions |

## 📁 Project Structure

```
delta_anti/
├── config.py                 # Configuration
├── main.py                   # Console dashboard entry
├── requirements.txt          # Dependencies
│
├── api/
│   ├── delta_rest.py        # REST API client
│   ├── delta_websocket.py   # WebSocket client
│   ├── server/              # FastAPI server
│   │   └── main.py
│   └── webhooks/            # Webhook handlers
│       └── tradingview.py
│
├── ml/                       # Machine Learning
│   ├── models/
│   │   └── lstm_predictor.py
│   ├── features/
│   │   └── feature_engineer.py
│   ├── sentiment/
│   │   └── market_sentiment.py
│   └── agents/
│       └── dqn_trader.py
│
├── data/
│   └── market_data.py
├── analysis/
│   ├── indicators.py
│   └── signals.py
├── strategy/
│   └── trade_manager.py
└── ui/
    └── dashboard.py
```

## 📊 Technical Indicators

| Indicator | Description |
|-----------|-------------|
| RSI | Relative Strength Index (14-period) |
| MACD | Moving Average Convergence Divergence |
| Bollinger Bands | 20-period with 2 std dev |
| ATR | Average True Range |
| ADX | Average Directional Index |
| EMA/SMA | Exponential & Simple Moving Averages |

## 🔗 API Documentation

- [Delta Exchange API Docs](https://docs.delta.exchange/)

## 📄 License

MIT
