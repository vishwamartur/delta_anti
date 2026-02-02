# Delta Anti - AI-Powered Trading System

A **production-ready** quantitative trading system for Delta Exchange with **AI/ML predictions**, **adaptive learning**, and **200x leverage support**.

## ✨ Features

### 🤖 AI/ML Trading Intelligence
- **Lag-Llama Forecaster**: Time series foundation model for price predictions
- **LSTM Predictor**: Bidirectional LSTM with attention mechanism
- **FinBERT Sentiment**: Financial news sentiment analysis
- **Adaptive Learning**: Learns from trade history to improve decisions

### 📈 Advanced Trading
- **200x Leverage Support**: Risk management optimized for high leverage
- **Limit Orders**: Uses maker orders (0.02% fees vs 0.05% taker)
- **Auto-Topup**: Prevents liquidation by adding margin automatically
- **Trailing Stops**: Activates after 0.5% profit to lock in gains
- **Trade Cooldown**: 60-second minimum between trades

### 📊 Technical Analysis
- **Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX, EMA/SMA
- **Signal Validation**: ML confirms technical signals before entry
- **Confidence Scores**: 0-100% confidence on every signal

### 🌐 REST API Server
- **FastAPI Server**: Production-ready endpoints
- **WebSocket Streaming**: Real-time predictions
- **Webhooks**: TradingView & Telegram integration

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/vishwamartur/delta_anti.git
cd delta_anti

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API credentials
cp .env.example .env
# Edit .env with your Delta Exchange API keys

# 4. (Optional) Install Lag-Llama for AI forecasting
python scripts/install_lag_llama.py

# 5. Start trading
python run_system.py
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```env
# Delta Exchange API
DELTA_API_KEY=your_api_key
DELTA_API_SECRET=your_api_secret
DELTA_REST_URL=https://api.india.delta.exchange
DELTA_WS_URL=wss://socket.india.delta.exchange

# Trading
TRADING_SYMBOLS=BTCUSD,ETHUSD
DEFAULT_TIMEFRAME=5m
RISK_PER_TRADE=0.10          # 10% risk per trade
DEFAULT_LEVERAGE=200
AUTO_TOPUP=true
AUTO_EXECUTION=true

# Trade Frequency
MIN_TRADE_INTERVAL=60        # Seconds between trades
```

### Key Settings Explained

| Setting | Default | Description |
|---------|---------|-------------|
| `RISK_PER_TRADE` | 0.10 | 10% of balance risked per trade |
| `DEFAULT_LEVERAGE` | 200 | Leverage multiplier |
| `AUTO_TOPUP` | true | Adds margin to prevent liquidation |
| `MIN_TRADE_INTERVAL` | 60 | Cooldown between trades (seconds) |

---

## 🤖 AI/ML Models

### 1. Lag-Llama (Foundation Model)
Time series transformer trained on 100+ datasets.

```bash
# Install (requires GPU, 4GB+ VRAM)
python scripts/install_lag_llama.py
```

**Logs:**
```
[LAG-LLAMA] BTCUSD: bullish (72%, +1.25%)
```

### 2. LSTM Predictor
Bidirectional LSTM with attention for price forecasting.

### 3. FinBERT Sentiment
Analyzes crypto news sentiment before trades.

```
[SENTIMENT] BTCUSD: bullish (score: 0.45)
```

### Model Fallback Chain
```
Lag-Llama → LSTM → Momentum → Technical Only
```

---

## 📁 Project Structure

```
delta_anti/
├── run_system.py              # Main entry point
├── config.py                  # Configuration
├── .env                       # API credentials
│
├── api/
│   ├── delta_rest.py          # REST API client
│   ├── delta_websocket.py     # WebSocket client
│   └── server/                # FastAPI server
│
├── ml/                        # Machine Learning
│   ├── models/
│   │   ├── lstm_predictor.py      # LSTM model
│   │   └── lag_llama_predictor.py # Lag-Llama model
│   ├── features/
│   │   └── feature_engineer.py    # 100+ features
│   └── sentiment/
│       └── market_sentiment.py    # FinBERT
│
├── analysis/
│   ├── indicators.py          # Technical indicators
│   └── signals.py             # Signal generation + ML validation
│
├── strategy/
│   ├── advanced_trade_manager.py  # Trade execution
│   └── trade_analyzer.py          # Adaptive learning
│
├── scripts/
│   └── install_lag_llama.py   # ML setup script
│
└── data/
    └── trades.json            # Trade history
```

---

## 📊 Trading Dashboard

When running, you'll see:

```
============================================================
[LIVE ENTRY] BTCUSD SHORT @ $78,500.00
  Order ID: 12345678
----------------------------------------
  Size:      1 contracts
  Notional:  $78,500.00
  Stop Loss: $79,128.00 (0.80% away)
  Take Profit: $77,558.00 (1.20% away)
  Risk/Reward: 1.5x
  Est. Fees:   $31.40 (round-trip @ 0.04%)
============================================================
```

---

## 💰 Fee Optimization

Uses **limit orders** to reduce fees:

| Fee Type | Market Order | Limit Order | Savings |
|----------|--------------|-------------|---------|
| Entry | 0.05% | 0.02% | 60% |
| Exit | 0.05% | 0.02% | 60% |
| **Round-trip** | 0.10% | **0.04%** | **60%** |

---

## 🛡️ Risk Management

1. **Daily Loss Limit**: 50% (configurable)
2. **Max Drawdown**: 70%
3. **Auto-Topup**: Uses wallet balance to prevent liquidation
4. **Trailing Stop**: Activates after 0.5% profit
5. **Trade Cooldown**: Prevents overtrading

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/trades/active` | GET | Get open trades |
| `/api/trades/stats` | GET | Get trading statistics |
| `/api/v2/predict` | POST | ML price predictions |
| `/api/v2/signals/{symbol}` | GET | Trading signals |
| `/api/v2/sentiment/{symbol}` | GET | Sentiment analysis |

**API Docs**: `http://localhost:8000/docs`

---

## 📈 Performance Tracking

Trade history saved to `data/trades.json`:

```json
{
  "stats": {
    "total_trades": 27,
    "win_rate": 40.0,
    "total_pnl": 125.50,
    "account_balance": 505.50
  }
}
```

---

## 🔧 Troubleshooting

### "Daily loss limit reached"
```bash
# Reset by restarting the bot
python run_system.py
```

### GPU not detected for Lag-Llama
```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### High fees
- Ensure `order_type` is `"maker"` in config
- Check that limit orders are being placed (see logs)

---

## 📄 License

MIT

---

## 🔗 Links

- [Delta Exchange API Docs](https://docs.delta.exchange/)
- [Lag-Llama Paper](https://arxiv.org/abs/2310.08278)
- [FinBERT](https://huggingface.co/ProsusAI/finbert)
