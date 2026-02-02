"""
Lag-Llama Installation Script
Run this to set up Lag-Llama time series forecaster.

Requirements:
- Python 3.8+
- CUDA-capable GPU (GTX 1650 Ti or better)
- ~4GB VRAM for inference
"""
import subprocess
import sys
import os


def run_cmd(cmd):
    """Run a command and print output."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True


def main():
    print("=" * 60)
    print("  LAG-LLAMA INSTALLATION FOR DELTA ANTI TRADING SYSTEM")
    print("=" * 60)
    
    # Step 1: Check Python version
    print("\n[1/5] Checking Python version...")
    if sys.version_info < (3, 8):
        print(f"Error: Python 3.8+ required, got {sys.version}")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Step 2: Check for CUDA
    print("\n[2/5] Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        else:
            print("⚠ No GPU detected. Lag-Llama will run on CPU (slower)")
    except ImportError:
        print("⚠ PyTorch not installed yet, will install with CUDA support")
    
    # Step 3: Install core dependencies
    print("\n[3/5] Installing dependencies...")
    dependencies = [
        "torch>=2.0.0",
        "gluonts>=0.14.0",
        "huggingface-hub",
        "lightning",
        "scipy",
        "pandas",
        "numpy"
    ]
    
    for dep in dependencies:
        if not run_cmd(f"{sys.executable} -m pip install {dep} --quiet"):
            print(f"Warning: Failed to install {dep}")
    
    # Step 4: Clone lag-llama repository
    print("\n[4/5] Cloning Lag-Llama repository...")
    lag_llama_dir = os.path.join(os.path.dirname(__file__), "..", "vendor", "lag-llama")
    
    if os.path.exists(lag_llama_dir):
        print(f"✓ Lag-Llama already exists at {lag_llama_dir}")
    else:
        os.makedirs(os.path.dirname(lag_llama_dir), exist_ok=True)
        if run_cmd(f"git clone https://github.com/time-series-foundation-models/lag-llama {lag_llama_dir}"):
            print("✓ Lag-Llama cloned successfully")
            # Install from cloned repo
            run_cmd(f"{sys.executable} -m pip install -e {lag_llama_dir} --quiet")
        else:
            print("⚠ Could not clone Lag-Llama. Will use fallback predictor.")
    
    # Step 5: Download model weights
    print("\n[5/5] Downloading Lag-Llama model weights...")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models", "lag_llama")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "lag-llama.ckpt")
    
    if os.path.exists(model_path):
        print(f"✓ Model already downloaded at {model_path}")
    else:
        try:
            from huggingface_hub import hf_hub_download
            print("Downloading from Hugging Face (this may take a few minutes)...")
            hf_hub_download(
                repo_id="time-series-foundation-models/Lag-Llama",
                filename="lag-llama.ckpt",
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            print(f"✓ Model downloaded to {models_dir}")
        except Exception as e:
            print(f"⚠ Could not download model: {e}")
            print("  You can manually download from: https://huggingface.co/time-series-foundation-models/Lag-Llama")
    
    # Done
    print("\n" + "=" * 60)
    print("  INSTALLATION COMPLETE!")
    print("=" * 60)
    print("\nLag-Llama is now ready to use.")
    print("\nTest with:")
    print("  python -c \"from ml.models.lag_llama_predictor import get_lag_llama_predictor; print('OK')\"")
    print("\nStart trading:")
    print("  python run_system.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
