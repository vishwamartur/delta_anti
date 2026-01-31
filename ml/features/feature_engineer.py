"""
Feature Engineering Pipeline
Creates 100+ features from raw OHLCV data for ML models
"""
import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """
    Advanced feature engineering for ML models.
    Creates comprehensive feature set from OHLCV data.
    """
    
    def __init__(self):
        self.feature_names: List[str] = []
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLCV data.
        
        Args:
            df: DataFrame with open, high, low, close, volume columns
            
        Returns:
            DataFrame with all engineered features
        """
        features = pd.DataFrame(index=df.index)
        
        # ========== Price Features ==========
        features['close'] = df['close']
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        
        # Price changes
        features['price_change'] = df['close'].diff()
        features['price_change_pct'] = df['close'].pct_change()
        features['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Candle patterns
        features['body'] = df['close'] - df['open']
        features['body_pct'] = features['body'] / df['open']
        features['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        features['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        features['range'] = df['high'] - df['low']
        features['range_pct'] = features['range'] / df['low']
        
        # ========== Moving Averages ==========
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # Price relative to MA
            features[f'price_sma_{period}_ratio'] = df['close'] / features[f'sma_{period}']
            features[f'price_ema_{period}_ratio'] = df['close'] / features[f'ema_{period}']
        
        # MA crossovers
        features['ema_9_21_cross'] = (
            features['ema_10'] - features['ema_20']
        ).apply(np.sign)
        features['ema_20_50_cross'] = (
            features['ema_20'] - features['ema_50']
        ).apply(np.sign)
        
        # ========== Momentum Indicators ==========
        
        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0).rolling(period).mean()
            rs = gain / loss.replace(0, np.inf)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic
        for period in [14, 21]:
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()
            features[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
            features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(3).mean()
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Rate of Change
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = (
                (df['close'] - df['close'].shift(period)) / 
                df['close'].shift(period)
            ) * 100
        
        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        # Williams %R
        for period in [14, 21]:
            high_max = df['high'].rolling(period).max()
            low_min = df['low'].rolling(period).min()
            features[f'williams_r_{period}'] = -100 * (
                (high_max - df['close']) / (high_max - low_min + 1e-10)
            )
        
        # ========== Volatility Indicators ==========
        
        # Standard deviation
        for period in [10, 20, 50]:
            features[f'std_{period}'] = df['close'].rolling(period).std()
            features[f'return_std_{period}'] = df['close'].pct_change().rolling(period).std()
        
        # ATR (Average True Range)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        for period in [7, 14, 21]:
            features[f'atr_{period}'] = true_range.rolling(period).mean()
            features[f'atr_pct_{period}'] = features[f'atr_{period}'] / df['close'] * 100
        
        # Bollinger Bands
        for period in [20]:
            for std_mult in [1, 2, 3]:
                mid = df['close'].rolling(period).mean()
                std = df['close'].rolling(period).std()
                features[f'bb_upper_{period}_{std_mult}'] = mid + std * std_mult
                features[f'bb_lower_{period}_{std_mult}'] = mid - std * std_mult
                features[f'bb_width_{period}_{std_mult}'] = (
                    features[f'bb_upper_{period}_{std_mult}'] - 
                    features[f'bb_lower_{period}_{std_mult}']
                ) / mid
                features[f'bb_position_{period}_{std_mult}'] = (
                    (df['close'] - features[f'bb_lower_{period}_{std_mult}']) /
                    (features[f'bb_upper_{period}_{std_mult}'] - 
                     features[f'bb_lower_{period}_{std_mult}'] + 1e-10)
                )
        
        # Keltner Channels
        ema_20 = df['close'].ewm(span=20).mean()
        atr_10 = true_range.rolling(10).mean()
        features['keltner_upper'] = ema_20 + atr_10 * 2
        features['keltner_lower'] = ema_20 - atr_10 * 2
        features['keltner_position'] = (
            (df['close'] - features['keltner_lower']) /
            (features['keltner_upper'] - features['keltner_lower'] + 1e-10)
        )
        
        # ========== Trend Indicators ==========
        
        # ADX (Average Directional Index)
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff().abs() * -1
        plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
        minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
        
        for period in [14]:
            plus_di = 100 * plus_dm.rolling(period).mean() / (features[f'atr_{period}'] + 1e-10)
            minus_di = 100 * minus_dm.rolling(period).mean() / (features[f'atr_{period}'] + 1e-10)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            features['adx'] = dx.rolling(period).mean()
            features['plus_di'] = plus_di
            features['minus_di'] = minus_di
        
        # ========== Volume Features ==========
        if 'volume' in df.columns:
            features['volume'] = df['volume']
            
            for period in [5, 10, 20, 50]:
                features[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
                features[f'volume_ratio_{period}'] = (
                    df['volume'] / features[f'volume_sma_{period}']
                )
            
            # On-Balance Volume (OBV)
            obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
            features['obv'] = obv
            features['obv_sma_20'] = obv.rolling(20).mean()
            
            # Volume-Price Trend (VPT)
            vpt = (df['close'].pct_change() * df['volume']).cumsum()
            features['vpt'] = vpt
            
            # Money Flow Index (MFI)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            
            for period in [14]:
                pos_mf = positive_flow.rolling(period).sum()
                neg_mf = negative_flow.rolling(period).sum()
                mfi = 100 * pos_mf / (pos_mf + neg_mf + 1e-10)
                features[f'mfi_{period}'] = mfi
        
        # ========== Statistical Features ==========
        
        # Z-Score
        for period in [20, 50]:
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'zscore_{period}'] = (df['close'] - mean) / (std + 1e-10)
        
        # Percentile rank
        features['percentile_50'] = df['close'].rolling(50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        features['percentile_100'] = df['close'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        
        # Skewness and Kurtosis
        features['returns_skew_20'] = df['close'].pct_change().rolling(20).skew()
        features['returns_kurt_20'] = df['close'].pct_change().rolling(20).kurt()
        
        # ========== Market Microstructure ==========
        
        # Bid-Ask spread proxy
        features['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Amihud illiquidity
        if 'volume' in df.columns:
            features['amihud'] = (
                abs(df['close'].pct_change()) / (df['volume'] + 1)
            )
        
        # ========== Fractal Features ==========
        
        # Hurst Exponent (simplified)
        features['hurst'] = self._calculate_hurst(df['close'])
        
        # ========== Time Features ==========
        if hasattr(df.index, 'hour'):
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Fill NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        self.feature_names = features.columns.tolist()
        
        return features
    
    def _calculate_hurst(self, ts: pd.Series, max_lag: int = 20) -> pd.Series:
        """
        Calculate rolling Hurst exponent for trend persistence.
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        def hurst_for_window(data):
            if len(data) < max_lag:
                return 0.5
            
            lags = range(2, max_lag)
            tau = []
            
            for lag in lags:
                pp = np.subtract(data[lag:], data[:-lag])
                tau.append(np.sqrt(np.std(pp)))
            
            if len(tau) == 0 or all(t == 0 for t in tau):
                return 0.5
            
            try:
                poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
                return poly[0]
            except:
                return 0.5
        
        return ts.rolling(100).apply(hurst_for_window, raw=True)
    
    def get_feature_importance(self, model, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from a trained model."""
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            })
            return importance.nlargest(top_n, 'importance')
        return pd.DataFrame()
    
    def select_features(self, df: pd.DataFrame, 
                        target: pd.Series, 
                        top_n: int = 50) -> List[str]:
        """
        Select top features using correlation analysis.
        """
        correlations = df.corrwith(target).abs()
        top_features = correlations.nlargest(top_n).index.tolist()
        return top_features


# Singleton instance
feature_engineer = FeatureEngineer()
