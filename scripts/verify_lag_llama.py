import sys
import os
import pandas as pd
import numpy as np
import logging

# Add project root
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY")

def verify_lag_llama():
    print("Verifying Lag-Llama Integration...")
    
    try:
        from ml.models.lag_llama_predictor import get_lag_llama_predictor
        print("Import Successful")
        
        predictor = get_lag_llama_predictor()
        print("Predictor Initialized. Attempting to load model explicitly...")
        
        try:
            predictor._initialize_model()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Model Init FAILED: {e}")
            # print stack trace
            import traceback
            traceback.print_exc()

        # Create dummy data
        dates = pd.date_range(start="2024-01-01", periods=200, freq="H")
        prices = np.linspace(100, 200, 200) + np.random.normal(0, 5, 200)
        df = pd.DataFrame({'close': prices, 'timestamp': dates})
        
        print("Running prediction...")
        result = predictor.predict(df, "TEST")
        
        if result:
            print(f"Prediction Success: {result.direction} ({result.confidence:.2f})")
            print(f"Forecast Horizon: {len(result.predicted_prices)} steps")
        else:
            print("Prediction returned None")
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_lag_llama()
