import torch
import os

CACHE_DIR = os.path.join(os.getcwd(), "models", "lag_llama")
MODEL_FILENAME = "lag-llama.ckpt"
model_path = os.path.join(CACHE_DIR, MODEL_FILENAME)

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    exit(1)

try:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model_kwargs = ckpt.get("hyper_parameters", {}).get("model_kwargs", {})
    lags = model_kwargs.get("lags_seq", None)
    
    print(f"Found lags_seq in checkpoint: {type(lags)}")
    if isinstance(lags, list):
        print(f"Length: {len(lags)}")
        print(f"First 10: {lags[:10]}")
        print(f"Full list: {lags}")
        
        # Check if they are ints or strings
        if len(lags) > 0:
            print(f"Item type: {type(lags[0])}")
            
except Exception as e:
    print(f"Error inspecting checkpoint: {e}")
