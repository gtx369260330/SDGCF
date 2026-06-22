"""Check whether the installed PyTorch can execute CUDA kernels on this GPU.

Run:
    python check_cuda_compatibility.py
"""
from __future__ import annotations

import torch

print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("Result: CUDA is not available. Use --device cpu, or install a CUDA-enabled PyTorch build.")
else:
    try:
        idx = torch.cuda.current_device()
        print("GPU count:", torch.cuda.device_count())
        print("Current GPU:", idx, torch.cuda.get_device_name(idx))
        print("Compute capability:", torch.cuda.get_device_capability(idx))
        x = torch.randn(32, 32, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("Result: CUDA kernel test passed. You can use --device cuda.")
    except Exception as exc:
        print("Result: CUDA is visible but kernel execution failed.")
        print("Error:", repr(exc))
        print("Recommended immediate fix: run with --device cpu.")
        print("For GPU acceleration: install a PyTorch CUDA version that supports this GPU's compute capability, or use a newer GPU.")
