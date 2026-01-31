# Delta Exchange Real-Time Trading System

A Python-based quantitative trading system for Delta Exchange that streams real-time market data, performs technical analysis, and provides trade entry/exit signals.

## Features

- **Real-Time Data**: WebSocket streaming of OHLC candles and tickers
- **10+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX, EMA/SMA
- **Trade Signals**: Long/Short entry with confidence scores (0-100)
- **Exit Tracking**: Dynamic TP/SL levels based on ATR
- **Risk Management**: Position sizing and trade limits
- **Console Dashboard**: Real-time display of all trading data

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`
2. Add your Delta Exchange API credentials:
   ```
   DELTA_API_KEY=your_api_key
   DELTA_API_SECRET=your_api_secret
   ```

> ⚠️ **Important**: Ensure your IP is whitelisted on Delta Exchange for API trading access.

## Usage

```bash
python main.py
```

This will launch the real-time dashboard showing:
- Live market prices with trend indicators
- Trade signals (LONG/SHORT) with entry, SL, and TP levels
- Open positions with P&L tracking
- Technical indicator values

Press `Ctrl+C` to stop.

## Project Structure

```
deltaanti/
├── .env                      # API credentials (not tracked)
├── config.py                 # Configuration settings
├── main.py                   # Application entry point
├── requirements.txt          # Python dependencies
├── api/
│   ├── delta_rest.py        # REST API client
│   └── delta_websocket.py   # WebSocket client
├── data/
│   └── market_data.py       # Candle data management
├── analysis/
│   ├── indicators.py        # Technical indicators
│   └── signals.py           # Signal generation
├── strategy/
│   └── trade_manager.py     # Trade tracking
└── ui/
    └── dashboard.py         # Console display
```

## Technical Indicators

| Indicator | Description |
|-----------|-------------|
| RSI | Relative Strength Index (14-period) |
| MACD | Moving Average Convergence Divergence |
| Bollinger Bands | 20-period with 2 std dev |
| ATR | Average True Range for volatility |
| ADX | Average Directional Index for trend strength |
| EMA/SMA | Exponential and Simple Moving Averages |

## API Documentation

Refer to the [Delta Exchange API Docs](https://docs.delta.exchange/) for more details.

## License

MIT
