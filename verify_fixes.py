import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

print("Verifying fixes...")

try:
    print("1. Importing lstm_predictor...")
    from ml.models.lstm_predictor import lstm_predictor
    print("   [OK] lstm_predictor imported successfully")
except Exception as e:
    print(f"   [FAIL] lstm_predictor import failed: {e}")

try:
    print("2. Importing LagLlamaEstimator...")
    # Add vendor to path for this check
    sys.path.append(os.path.join(os.getcwd(), 'vendor', 'lag-llama'))
    from lag_llama.gluon.estimator import LagLlamaEstimator
    
    # Try initializing to trigger the frequency check
    estimator = LagLlamaEstimator(
        prediction_length=24,
        context_length=32,
        input_size=1,
        n_layer=1,
        n_embd_per_head=32,
        n_head=4
    )
    print("   [OK] LagLlamaEstimator initialized successfully")
except Exception as e:
    print(f"   [FAIL] LagLlamaEstimator failed: {e}")

print("Verification complete.")
