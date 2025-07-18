#!/bin/bash
# Start Kimi-K2 KTransformers server with CPU+AMX optimization

echo "Starting Kimi-K2 CPU+AMX Server..."

# Check if model exists
MODEL_PATH="/tmp/kimi-ramdisk/Kimi-K2-Instruct"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please ensure Kimi-K2 model is loaded in RAM disk"
    exit 1
fi

# Set environment variables for CPU optimization
export OMP_NUM_THREADS=192
export MKL_NUM_THREADS=192
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export KMP_BLOCKTIME=0
export MKL_DYNAMIC=false

# NUMA optimization
export OMP_WAIT_POLICY=ACTIVE
export GOMP_CPU_AFFINITY="0-191"

# PyTorch settings
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TORCH_USE_CUDA_DSA=0

# Memory settings
export MALLOC_ARENA_MAX=2
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072

# Check if ktransformers is installed
if [ ! -d "/root/ktransformers" ]; then
    echo "Error: KTransformers not found at /root/ktransformers"
    echo "Please install KTransformers first"
    exit 1
fi

# Navigate to ktransformers directory
cd /root/ktransformers

# Start the server
echo "Launching KTransformers server..."
echo "Configuration: CPU+AMX with INT8 quantization"
echo "Model: Kimi-K2-Instruct (1T parameters, 32B active)"
echo "Memory: 2TB RAM with model pre-loaded"
echo "CPUs: 192 cores (Intel Xeon Platinum 8468)"
echo ""

# Run server with our custom config
python -m ktransformers.server \
    --config /root/kimi_k2_cpu_amx_config.yaml \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info

# Note: Using single worker to avoid memory duplication
# Each worker would need full model in memory