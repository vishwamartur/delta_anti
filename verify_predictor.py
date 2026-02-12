import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

print("Verifying LagLlamaPredictor initialization...")

try:
    from ml.models.lag_llama_predictor import LagLlamaPredictor
    
    # Initialize predictor (which attempts to load model and create estimator)
    # We use CPU to avoid VRAM issues during test if possible, or auto
    predictor = LagLlamaPredictor(device="cpu")
    
    print("Initializing model...")
    predictor._initialize_model()
    
    if predictor.is_initialized:
        print("   [OK] LagLlamaPredictor initialized successfully")
    else:
        print("   [FAIL] LagLlamaPredictor failed to initialize (is_initialized=False)")

except Exception as e:
    print(f"   [FAIL] LagLlamaPredictor initialization crashed: {e}")
    import traceback
    traceback.print_exc()

print("Verification complete.")
