import sys
import os
# Ensure vendor dir is in path
vendor_dir = os.path.join(os.getcwd(), "vendor", "lag-llama")
sys.path.append(vendor_dir)

import inspect
try:
    from lag_llama.gluon.estimator import LagLlamaEstimator
    print("Imported LagLlamaEstimator successfully.")
    
    # Inspect arguments
    sig = inspect.signature(LagLlamaEstimator.__init__)
    print("LagLlamaEstimator.__init__ parameters:")
    for param in sig.parameters:
        print(f"  {param}")
        
    print(f"\nFile: {inspect.getfile(LagLlamaEstimator)}")
    
except Exception as e:
    print(f"Error importing: {e}")
    import traceback
    traceback.print_exc()
