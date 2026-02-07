"""
LSTM Training Script - 5 Year Historical Data Training
Trains the native AI model on 5 years of historical BTC/ETH data for strong trading predictions.

Usage:
    python train_lstm.py [--symbols BTCUSD ETHUSD] [--years 5] [--epochs 200]
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
from api.delta_rest import rest_client
from ml.models.lstm_predictor import LSTMPredictor
import config


class HistoricalDataFetcher:
    """Fetches historical OHLCV data from Delta Exchange and external sources."""
    
    def __init__(self):
        self.client = rest_client
    
    def fetch_delta_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Fetch historical data from Delta Exchange API (limited history)."""
        logger.info(f"[DATA] Fetching {days} days of {symbol} data from Delta Exchange...")
        
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=days)).timestamp())
        
        all_data = []
        current_end = end_time
        
        # Delta API returns data in chunks, need to paginate
        while current_end > start_time:
            try:
                response = self.client.get_candles(
                    symbol=symbol,
                    resolution="1d",  # Daily candles for 5 years
                    start=start_time,
                    end=current_end
                )
                
                if 'result' in response and response['result']:
                    candles = response['result']
                    if not candles:
                        break
                    
                    all_data.extend(candles)
                    
                    # Move to earlier period
                    earliest = min(c.get('time', c.get('t', current_end)) for c in candles)
                    if earliest >= current_end:
                        break
                    current_end = earliest - 1
                    
                    time.sleep(0.5)  # Rate limiting
                else:
                    break
                    
            except Exception as e:
                logger.warning(f"[DATA] Error fetching: {e}")
                break
        
        if all_data:
            df = self._process_candles(all_data)
            logger.info(f"[DATA] Fetched {len(df)} candles from Delta Exchange")
            return df
        
        return pd.DataFrame()
    
    def fetch_external_data(self, symbol: str, years: int = 5) -> pd.DataFrame:
        """Fetch historical data using yfinance for free 5-year+ data."""
        logger.info(f"[DATA] Fetching {years} years of {symbol} data using yfinance...")
        
        # Map symbols to Yahoo Finance tickers
        ticker_map = {
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD',
            'BTCUSDT': 'BTC-USD',
            'ETHUSDT': 'ETH-USD'
        }
        
        ticker = ticker_map.get(symbol, 'BTC-USD')
        
        try:
            import yfinance as yf
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years)
            
            logger.info(f"[DATA] Downloading {ticker} from {start_date.date()} to {end_date.date()}")
            
            # Download data
            data = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d',
                progress=False
            )
            
            if data.empty:
                logger.warning(f"[DATA] No data from yfinance for {ticker}")
                return pd.DataFrame()
            
            logger.info(f"[DATA] Raw data shape: {data.shape}, columns: {list(data.columns)}")
            
            # Handle multi-level columns from yfinance 1.1.0
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten multi-level columns
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            
            # Reset index to get Date as a column
            df = data.reset_index()
            
            # Standardize column names (lowercase)
            df.columns = [str(c).lower().strip() for c in df.columns]
            
            # Rename to expected format
            col_mapping = {
                'date': 'timestamp',
                'datetime': 'timestamp',
                'index': 'timestamp',
                'adj close': 'adj_close'
            }
            df = df.rename(columns=col_mapping)
            
            # Select only needed columns
            needed_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [c for c in needed_cols if c in df.columns]
            
            if len(available_cols) < 5:
                logger.warning(f"[DATA] Missing columns. Available: {list(df.columns)}")
                return pd.DataFrame()
            
            df = df[available_cols]
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            df.attrs['symbol'] = symbol
            
            logger.info(f"[DATA] ✓ Fetched {len(df)} days from yfinance ({years} years)")
            return df
            
        except Exception as e:
            logger.error(f"[DATA] yfinance data fetch failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _process_candles(self, candles: list) -> pd.DataFrame:
        """Process raw candle data to DataFrame."""
        processed = []
        for c in candles:
            processed.append({
                'timestamp': pd.to_datetime(c.get('time', c.get('t', 0)), unit='s'),
                'open': float(c.get('open', c.get('o', 0))),
                'high': float(c.get('high', c.get('h', 0))),
                'low': float(c.get('low', c.get('l', 0))),
                'close': float(c.get('close', c.get('c', 0))),
                'volume': float(c.get('volume', c.get('v', 0)))
            })
        
        df = pd.DataFrame(processed)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset='timestamp')
        return df
    
    def fetch_combined_data(self, symbol: str, years: int = 5) -> pd.DataFrame:
        """Combine Delta Exchange recent data with external historical data."""
        # Get long-term history from external source
        external_df = self.fetch_external_data(symbol, years)
        
        # Get recent high-quality data from Delta
        delta_df = self.fetch_delta_data(symbol, days=90)
        
        if external_df.empty:
            logger.warning(f"[DATA] Using only Delta data for {symbol}")
            return delta_df
        
        if delta_df.empty:
            logger.warning(f"[DATA] Using only external data for {symbol}")
            external_df.attrs['symbol'] = symbol
            return external_df
        
        # Combine: use external for history, Delta for recent
        # Remove overlapping period from external
        cutoff = delta_df['timestamp'].min()
        external_df = external_df[external_df['timestamp'] < cutoff]
        
        combined = pd.concat([external_df, delta_df], ignore_index=True)
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        combined.attrs['symbol'] = symbol
        
        logger.info(f"[DATA] Combined dataset: {len(combined)} total records")
        return combined


class LSTMTrainer:
    """Manages LSTM training with 5-year historical data."""
    
    def __init__(self):
        self.data_fetcher = HistoricalDataFetcher()
        self.predictor = LSTMPredictor(
            sequence_length=60,
            prediction_steps=5,
            features=18  # Matches prepare_features() output
        )
        self.training_stats = {}
    
    def train_on_symbol(self, symbol: str, years: int = 5, epochs: int = 200) -> dict:
        """Train LSTM on historical data for a single symbol."""
        logger.info(f"\n{'='*60}")
        logger.info(f"[TRAIN] Starting training for {symbol} ({years} years)")
        logger.info(f"{'='*60}")
        
        # Fetch historical data
        df = self.data_fetcher.fetch_combined_data(symbol, years)
        
        if df.empty or len(df) < 100:
            logger.error(f"[TRAIN] Insufficient data for {symbol}: {len(df)} records")
            return {'success': False, 'error': 'Insufficient data'}
        
        logger.info(f"[TRAIN] Training on {len(df)} data points spanning "
                   f"{(df['timestamp'].max() - df['timestamp'].min()).days} days")
        
        # Train the model
        start_time = time.time()
        self.predictor.train(df, epochs=epochs, batch_size=32, validation_split=0.2)
        training_time = time.time() - start_time
        
        # Calculate training quality metrics
        quality_metrics = self._evaluate_training_quality()
        
        # Save the trained model
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f'lstm_{symbol.lower()}.pkl')
        self.predictor.save(model_path)
        
        # Also save as default model
        default_path = os.path.join(model_dir, 'lstm_predictor.pkl')
        self.predictor.save(default_path)
        
        stats = {
            'success': True,
            'symbol': symbol,
            'data_points': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'training_time_seconds': round(training_time, 2),
            'epochs_completed': len(self.predictor.training_history),
            'final_train_loss': self.predictor.training_history[-1]['train_loss'] if self.predictor.training_history else None,
            'final_val_loss': self.predictor.training_history[-1]['val_loss'] if self.predictor.training_history else None,
            'model_path': model_path,
            'quality_metrics': quality_metrics
        }
        
        self.training_stats[symbol] = stats
        
        logger.info(f"\n[TRAIN] ✓ Training complete for {symbol}")
        logger.info(f"[TRAIN]   Data: {len(df)} points, {(df['timestamp'].max() - df['timestamp'].min()).days} days")
        logger.info(f"[TRAIN]   Time: {training_time:.1f}s")
        logger.info(f"[TRAIN]   Final Val Loss: {stats['final_val_loss']:.6f}")
        logger.info(f"[TRAIN]   Model saved: {model_path}")
        
        return stats
    
    def _evaluate_training_quality(self) -> dict:
        """Evaluate the quality of training for trade decisions."""
        if not self.predictor.training_history:
            return {'score': 0, 'rating': 'UNTRAINED'}
        
        history = self.predictor.training_history
        final_val_loss = history[-1]['val_loss']
        
        # Calculate trend of validation loss
        recent_losses = [h['val_loss'] for h in history[-10:]]
        loss_trend = (recent_losses[-1] - recent_losses[0]) / recent_losses[0] if recent_losses[0] > 0 else 0
        
        # Score based on final loss and convergence
        if final_val_loss < 0.001:
            score = 95
            rating = 'EXCELLENT'
        elif final_val_loss < 0.005:
            score = 85
            rating = 'STRONG'
        elif final_val_loss < 0.01:
            score = 75
            rating = 'GOOD'
        elif final_val_loss < 0.02:
            score = 65
            rating = 'MODERATE'
        else:
            score = 50
            rating = 'WEAK'
        
        # Adjust for convergence stability
        if loss_trend < -0.1:  # Still improving
            score = min(100, score + 5)
        elif loss_trend > 0.1:  # Diverging
            score = max(30, score - 10)
            rating = 'UNSTABLE'
        
        return {
            'score': score,
            'rating': rating,
            'final_loss': final_val_loss,
            'convergence_trend': 'improving' if loss_trend < 0 else 'stable' if abs(loss_trend) < 0.1 else 'diverging'
        }
    
    def train_all_symbols(self, symbols: list, years: int = 5, epochs: int = 200):
        """Train on all trading symbols."""
        logger.info(f"\n{'='*60}")
        logger.info(f"[TRAIN] LSTM 5-Year Historical Training")
        logger.info(f"[TRAIN] Symbols: {', '.join(symbols)}")
        logger.info(f"[TRAIN] Years: {years}, Epochs: {epochs}")
        logger.info(f"{'='*60}\n")
        
        results = {}
        for symbol in symbols:
            result = self.train_on_symbol(symbol, years, epochs)
            results[symbol] = result
            
            # Save training summary
            self._save_training_summary(results)
        
        # Print final summary
        self._print_summary(results)
        
        return results
    
    def _save_training_summary(self, results: dict):
        """Save training summary to file."""
        import json
        
        summary_path = os.path.join(
            os.path.dirname(__file__), 
            'models', 
            'training_summary.json'
        )
        
        summary = {
            'last_trained': datetime.now().isoformat(),
            'results': results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _print_summary(self, results: dict):
        """Print training summary."""
        logger.info(f"\n{'='*60}")
        logger.info("[TRAIN] TRAINING SUMMARY")
        logger.info(f"{'='*60}")
        
        for symbol, stats in results.items():
            if stats.get('success'):
                quality = stats.get('quality_metrics', {})
                logger.info(f"\n  {symbol}:")
                logger.info(f"    ✓ Data: {stats['data_points']} points")
                logger.info(f"    ✓ Loss: {stats['final_val_loss']:.6f}")
                logger.info(f"    ✓ Quality: {quality.get('rating', 'N/A')} ({quality.get('score', 0)}%)")
                logger.info(f"    ✓ Model: {stats['model_path']}")
            else:
                logger.info(f"\n  {symbol}:")
                logger.info(f"    ✗ Failed: {stats.get('error', 'Unknown error')}")
        
        logger.info(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train LSTM on 5 years of historical data')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSD', 'ETHUSD'],
                        help='Trading symbols to train on')
    parser.add_argument('--years', type=int, default=5,
                        help='Years of historical data to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs')
    
    args = parser.parse_args()
    
    trainer = LSTMTrainer()
    trainer.train_all_symbols(args.symbols, args.years, args.epochs)


if __name__ == '__main__':
    main()
