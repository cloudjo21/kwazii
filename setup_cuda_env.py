#!/usr/bin/env python3
"""
CUDA Environment Setup for PyTorch with cuDNN

This module sets up the necessary environment variables to resolve
the libcudnn.so.9 library loading issue when using PyTorch with CUDA.

Usage:
    import setup_cuda_env  # Import this at the beginning of your notebook
    
Or run as script:
    uv run python setup_cuda_env.py
"""

import os
import sys
from pathlib import Path


def setup_cudnn_library_path():
    """Set up the LD_LIBRARY_PATH for cuDNN libraries."""
    # Get the virtual environment path
    venv_path = Path(sys.prefix)
    cudnn_lib_path = venv_path / 'lib' / 'python3.13' / 'site-packages' / 'nvidia' / 'cudnn' / 'lib'

    if cudnn_lib_path.exists():
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        cudnn_path_str = str(cudnn_lib_path)

        if cudnn_path_str not in current_ld_path:
            new_ld_path = f"{cudnn_path_str}:{current_ld_path}" if current_ld_path else cudnn_path_str
            os.environ['LD_LIBRARY_PATH'] = new_ld_path
            print(f"✅ Added cuDNN library path: {cudnn_path_str}")
        else:
            print("✅ cuDNN library path already set")

        return True
    else:
        print(f"⚠️  cuDNN library path not found: {cudnn_lib_path}")
        return False


def verify_torch_cuda():
    """Verify that PyTorch can access CUDA with cuDNN."""
    try:
        import torch
        print(f"📦 PyTorch version: {torch.__version__}")
        print(f"🔥 CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"🎯 CUDA version: {torch.version.cuda}")
            print(f"📱 Device name: {torch.cuda.get_device_name(0)}")
            print(f"🧠 cuDNN version: {torch.backends.cudnn.version()}")

            # Test tensor creation on GPU
            x = torch.randn(2, 3).cuda()
            print(f"✅ GPU tensor test: {x.device}")
            return True
        else:
            print("⚠️  CUDA not available")
            return False

    except ImportError as e:
        print(f"❌ PyTorch import error: {e}")
        return False
    except Exception as e:
        print(f"❌ CUDA test error: {e}")
        return False


def main():
    """Main function to set up environment and verify."""
    print("🚀 Setting up CUDA environment for PyTorch...")

    # Set up cuDNN library path
    cudnn_setup = setup_cudnn_library_path()

    # Verify PyTorch CUDA functionality
    if cudnn_setup:
        torch_success = verify_torch_cuda()
        if torch_success:
            print(
                "🎉 Environment setup complete! PyTorch with CUDA is ready to use."
            )
        else:
            print(
                "⚠️  Environment setup completed but PyTorch CUDA verification failed."
            )
    else:
        print(
            "❌ Environment setup failed. Please check your PyTorch installation."
        )


# Automatically run setup when imported
if __name__ == "__main__":
    main()
else:
    # When imported as module, automatically set up the environment
    setup_cudnn_library_path()
