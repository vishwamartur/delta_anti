import sys
import os
import torch
import logging

# Suppress logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Path setup
vendor_dir = os.path.join(os.getcwd(), "vendor", "lag-llama")
sys.path.append(vendor_dir)

try:
    from lag_llama.gluon.estimator import LagLlamaEstimator
    from lag_llama.gluon.lightning_module import LagLlamaLightningModule
    
    print("Modules imported.")
    
    # Mock parameters
    ckpt_path = "lag-llama.ckpt" # Assuming it exists in root from previous run
    if not os.path.exists(ckpt_path):
        print(f"Ckpt not found at {ckpt_path}")
        # sys.exit(0) # Don't exit, try init without ckpt first?
    
    # Try init Estimator
    try:
        est = LagLlamaEstimator(
            prediction_length=24,
            context_length=32,
            input_size=1,
            n_layer=2,
            n_embd_per_head=32,
            n_head=2,
            ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None
        )
        print("Estimator instantiated.")
        
        # Try creating predictor (which triggers load_from_checkpoint)
        # We need a dummy transformation and module?
        # Actually, let's try to trigger the failure point.
        # lag_llama_predictor.py calls _initialize_model which calls estimator.create_predictor(...) ??
        # No, _initialize_model calls estimator.create_lightning_module() indirectly?
        # No, LagLlamaPredictor._initialize_model:
        #   self.predictor = self.estimator.create_predictor(...)
        
        # self.estimator.create_predictor calls self.create_lightning_module()
        
        print("Attempting to create lightning module...")
        module = est.create_lightning_module()
        print("Lightning module created successfully.")

    except TypeError as te:
        print(f"\nCaught Expected TypeError: {te}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\nCaught unexpected exception: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"Import failed: {e}")
