
import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, filename='debug_output.txt', filemode='w')
logger = logging.getLogger(__name__)

from ml.models.lag_llama_predictor import LagLlamaPredictor

def create_dummy_data(length=300):
    dates = [datetime.now() - timedelta(hours=i) for i in range(length)]
    dates.reverse()
    
    # Create simple sine wave price
    prices = 50000 + 1000 * np.sin(np.linspace(0, 10, length)) + np.random.normal(0, 100, length)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'volume': 1000
    })
    return df.set_index('timestamp')

def log_print(msg):
    print(msg)
    logging.info(msg)

def main():
    log_print("Initializing LagLlamaPredictor...")
    try:
        # Use auto device to match production behavior
        predictor = LagLlamaPredictor(device="auto") 
        
        log_print("Creating dummy data...")
        df = create_dummy_data()
        
        log_print("Running prediction...")
        result = predictor.predict(df, symbol="BTCUSD")
        
        if result:
            log_print("Prediction successful!")
            log_print(f"Direction: {result.direction}")
            log_print(f"Confidence: {result.confidence}")
            log_print(f"Predicted Prices: {result.predicted_prices[:5]}...")
        else:
            log_print("No prediction returned.")
            
    except Exception as e:
        log_print(f"\nCaught exception during execution:")
        log_print(f"{type(e).__name__}: {e}")
        traceback_str = traceback.format_exc()
        log_print(traceback_str)

if __name__ == "__main__":
    main()
