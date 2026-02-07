"""
LSTM Price Predictor with Attention Mechanism
Predicts future price movements for trading signals
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from dataclasses import dataclass
import pickle
import os

# Try importing torch, fall back to numpy-based model if not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[ML] PyTorch not installed. Using simplified numpy-based predictor.")

from sklearn.preprocessing import MinMaxScaler


@dataclass
class PredictionResult:
    """Container for prediction results."""
    symbol: str
    current_price: float
    predicted_prices: List[float]
    direction: str  # "UP", "DOWN", "NEUTRAL"
    confidence: float  # 0-100
    predicted_change_pct: float


if TORCH_AVAILABLE:
    class AttentionLayer(nn.Module):
        """Self-attention layer for sequence processing."""
        
        def __init__(self, hidden_size: int, num_heads: int = 4):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                hidden_size, num_heads, batch_first=True
            )
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        def forward(self, x):
            attn_out, _ = self.attention(x, x, x)
            return self.layer_norm(x + attn_out)


    class LSTMPriceModel(nn.Module):
        """
        LSTM model with attention for price prediction.
        
        Architecture:
        - Bidirectional LSTM layers
        - Self-attention mechanism
        - Dropout regularization
        - Dense output layers
        """
        
        def __init__(self, input_size: int = 20, hidden_size: int = 128,
                     num_layers: int = 2, output_size: int = 5, dropout: float = 0.2):
            super().__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # Bidirectional LSTM
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout, bidirectional=True
            )
            
            # Attention layer (hidden_size * 2 for bidirectional)
            self.attention = AttentionLayer(hidden_size * 2)
            
            # Output layers
            self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            # LSTM processing
            lstm_out, _ = self.lstm(x)
            
            # Apply attention
            attn_out = self.attention(lstm_out)
            
            # Take the last time step
            out = attn_out[:, -1, :]
            
            # Dense layers
            out = self.dropout(self.relu(self.fc1(out)))
            predictions = self.fc2(out)
            
            return predictions


class LSTMPredictor:
    """
    High-level LSTM predictor for price forecasting.
    
    Features:
    - Automatic data preprocessing
    - Multi-step price prediction
    - Confidence scoring
    - Direction classification
    """
    
    def __init__(self, sequence_length: int = 60, prediction_steps: int = 5,
                 features: int = 20):
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.features = features
        
        # Scalers
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        # Model
        if TORCH_AVAILABLE:
            self.model = LSTMPriceModel(
                input_size=features,
                output_size=prediction_steps
            )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
        else:
            self.model = None
        
        self.is_trained = False
        self.training_history = []
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix from OHLCV data.
        Creates technical features for model input.
        """
        features = pd.DataFrame()
        
        # Price features
        features['close'] = df['close']
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Volatility
        features['volatility_10'] = df['close'].pct_change().rolling(10).std()
        features['volatility_20'] = df['close'].pct_change().rolling(20).std()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.inf)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # Bollinger position
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        features['bb_position'] = (df['close'] - bb_mid) / (2 * bb_std)
        
        # Volume
        if 'volume' in df.columns:
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Fill NaN values
        features = features.bfill().fillna(0)
        
        return features.values[:, :self.features]
    
    def create_sequences(self, data: np.ndarray, 
                         targets: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_steps + 1):
            X.append(data[i:i + self.sequence_length])
            if targets is not None:
                y.append(targets[i + self.sequence_length:
                                i + self.sequence_length + self.prediction_steps])
        
        return np.array(X), np.array(y) if targets is not None else None
    
    def train(self, df: pd.DataFrame, epochs: int = 100, 
              batch_size: int = 32, validation_split: float = 0.2):
        """
        Train the LSTM model on historical data.
        
        Args:
            df: DataFrame with OHLCV data
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
        """
        if not TORCH_AVAILABLE:
            print("[ML] PyTorch not available. Cannot train model.")
            return
        
        print(f"[ML] Preparing training data...")
        
        # Prepare features
        features = self.prepare_features(df)
        targets = df['close'].values.reshape(-1, 1)
        
        # Scale data
        features_scaled = self.feature_scaler.fit_transform(features)
        targets_scaled = self.target_scaler.fit_transform(targets)
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, targets_scaled.flatten())
        
        # Train/validation split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        
        print(f"[ML] Training on {len(X_train)} samples, validating on {len(X_val)}...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            self.model.train()
            
            # Mini-batch training
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            scheduler.step(val_loss)
            
            avg_train_loss = total_loss / n_batches
            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_loss
            })
            
            if (epoch + 1) % 10 == 0:
                print(f"[ML] Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_best_model()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[ML] Early stopping at epoch {epoch + 1}")
                    break
        
        self.is_trained = True
        print(f"[ML] Training complete! Best val loss: {best_val_loss:.6f}")
    
    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """
        Make price predictions on new data.
        
        Args:
            df: DataFrame with recent OHLCV data
            
        Returns:
            PredictionResult with predictions and confidence
        """
        if not TORCH_AVAILABLE or not self.is_trained:
            # Fallback: Simple momentum-based prediction
            return self._fallback_prediction(df)
        
        # Prepare features
        features = self.prepare_features(df)
        features_scaled = self.feature_scaler.transform(features)
        
        # Get last sequence
        if len(features_scaled) < self.sequence_length:
            return self._fallback_prediction(df)
        
        sequence = features_scaled[-self.sequence_length:]
        sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(sequence).cpu().numpy()[0]
        
        # Inverse transform
        predictions = self.target_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        
        current_price = df['close'].iloc[-1]
        
        # Calculate metrics
        predicted_change_pct = ((predictions[-1] - current_price) / current_price) * 100
        
        # Determine direction
        if predicted_change_pct > 0.5:
            direction = "UP"
        elif predicted_change_pct < -0.5:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"
        
        # Calculate confidence based on prediction consistency
        price_diffs = np.diff(predictions)
        consistency = np.sum(price_diffs > 0) / len(price_diffs) if direction == "UP" else \
                     np.sum(price_diffs < 0) / len(price_diffs) if direction == "DOWN" else 0.5
        confidence = min(95, 50 + consistency * 50)
        
        return PredictionResult(
            symbol=df.attrs.get('symbol', 'UNKNOWN'),
            current_price=current_price,
            predicted_prices=predictions.tolist(),
            direction=direction,
            confidence=confidence,
            predicted_change_pct=predicted_change_pct
        )
    
    def _fallback_prediction(self, df: pd.DataFrame) -> PredictionResult:
        """Simple momentum-based fallback when ML not available."""
        current_price = df['close'].iloc[-1]
        
        # Use simple momentum
        momentum = (current_price - df['close'].iloc[-20]) / df['close'].iloc[-20]
        
        # Project based on recent trend
        predictions = []
        for i in range(1, self.prediction_steps + 1):
            pred = current_price * (1 + momentum * 0.1 * i)
            predictions.append(pred)
        
        predicted_change_pct = ((predictions[-1] - current_price) / current_price) * 100
        
        if predicted_change_pct > 0.3:
            direction = "UP"
        elif predicted_change_pct < -0.3:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"
        
        return PredictionResult(
            symbol=df.attrs.get('symbol', 'UNKNOWN'),
            current_price=current_price,
            predicted_prices=predictions,
            direction=direction,
            confidence=55,  # Lower confidence for fallback
            predicted_change_pct=predicted_change_pct
        )
    
    def _save_best_model(self):
        """Save the current model state."""
        if TORCH_AVAILABLE:
            os.makedirs('models', exist_ok=True)
            torch.save(self.model.state_dict(), 'models/lstm_best.pth')
    
    def save(self, path: str = 'models/lstm_predictor.pkl'):
        """Save the complete predictor."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            'sequence_length': self.sequence_length,
            'prediction_steps': self.prediction_steps,
            'features': self.features,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        if TORCH_AVAILABLE and self.is_trained:
            torch.save(self.model.state_dict(), path.replace('.pkl', '_model.pth'))
        
        print(f"[ML] Predictor saved to {path}")
    
    def load(self, path: str = 'models/lstm_predictor.pkl'):
        """Load a saved predictor."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.sequence_length = state['sequence_length']
        self.prediction_steps = state['prediction_steps']
        self.features = state['features']
        self.feature_scaler = state['feature_scaler']
        self.target_scaler = state['target_scaler']
        self.is_trained = state['is_trained']
        self.training_history = state['training_history']
        
        if TORCH_AVAILABLE:
            model_path = path.replace('.pkl', '_model.pth')
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        print(f"[ML] Predictor loaded from {path}")


def _create_lstm_predictor():
    """Create LSTM predictor and auto-load trained model if available."""
    predictor = LSTMPredictor(features=18)
    
    # Auto-load trained model if it exists
    model_paths = [
        'models/lstm_predictor.pkl',  # Default trained model
        'models/lstm_btcusd.pkl',     # BTC-specific model
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                predictor.load(path)
                print(f"[ML] ✓ Auto-loaded trained LSTM from {path}")
                print(f"[ML]   Trained: {predictor.is_trained}, "
                      f"History: {len(predictor.training_history)} epochs")
                break
            except Exception as e:
                print(f"[ML] ⚠ Failed to load {path}: {e}")
                continue
    else:
        print("[ML] ℹ No trained LSTM model found. Run train_lstm.py to train.")
    
    return predictor


# Singleton instance - auto-loads trained model on import
lstm_predictor = _create_lstm_predictor()
