import os
import torch
import sys

print("-" * 60)
print("SLURM GPU DEBUGGER")
print("-" * 60)

# 1. Check Important Environment Variables
# Slurm sets these to tell the application which GPU to use
print(f"Host Name:            {os.environ.get('HOSTNAME', 'Unknown')}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
print(f"SLURM_JOB_GPUS:       {os.environ.get('SLURM_JOB_GPUS', 'NOT SET')}")
print(f"SLURM_GPUS_ON_NODE:   {os.environ.get('SLURM_GPUS_ON_NODE', 'NOT SET')}")

print("-" * 60)

# 2. Check PyTorch Internals
print(f"PyTorch Version:      {torch.__version__}")
# This matches the CUDA version PyTorch was compiled with
print(f"PyTorch CUDA Arch:    {torch.version.cuda}") 
print(f"Is CUDA available?    {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Device Count:         {device_count}")
    for i in range(device_count):
        print(f" > Device {i}:          {torch.cuda.get_device_name(i)}")
        
    # 3. Test Actual Allocation
    try:
        x = torch.tensor([1.0, 2.0]).cuda()
        print(f"\n SUCCESS: Tensor allocated on GPU: {x}")
    except Exception as e:
        print(f"\n FAILURE: Could not allocate tensor. Error: {e}")
else:
    print("\n FATAL: PyTorch sees NO GPUs.")
    # Common check: Did the user install the CPU-only version by mistake?
    if "+cpu" in torch.__version__:
        print("   >> WARNING: You seem to have installed the CPU-only version of PyTorch.")

print("-" * 60)