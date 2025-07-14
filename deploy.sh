#\!/bin/bash
set -e

echo "🚀 Kimi-K2 Deployment Script"
echo "============================="

# Check system requirements
echo "📋 Checking system requirements..."
python3 --version
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "Warning: No GPU detected"
free -h

# Set environment variables
export CUDA_HOME=/usr/local/cuda-12.2
export TORCH_CUDA_ARCH_LIST="9.0"
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH

# Create directories
mkdir -p models logs

# Check if models exist
if [ \! -d "./kimi-k2-instruct" ]; then
    echo "❌ Model not found. Please download first:"
    echo "huggingface-cli download moonshotai/Kimi-K2-Instruct --local-dir ./kimi-k2-instruct"
    exit 1
fi

if [ \! -d "./kimi-k2-gguf" ]; then
    echo "❌ GGUF model not found. Please download first:"
    echo "huggingface-cli download KVCache-ai/Kimi-K2-Instruct-GGUF --local-dir ./kimi-k2-gguf"
    exit 1
fi

# Install KTransformers if not present
if [ \! -d "./ktransformers" ]; then
    echo "📦 Installing KTransformers..."
    git clone https://github.com/kvcache-ai/ktransformers.git
    cd ktransformers
    pip install -e . --no-deps --no-build-isolation
    cd ..
fi

# Start server
echo "🌐 Starting Kimi-K2 server on port 10002..."
cd ktransformers
python3 -m ktransformers.server.main \
    --port 10002 \
    --model_path ../kimi-k2-instruct \
    --gguf_path ../kimi-k2-gguf \
    --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml \
    --max_new_tokens 512 \
    --cache_lens 4096 \
    --chunk_size 64 \
    --max_batch_size 1 \
    --backend_type ktransformers \
    2>&1 | tee ../logs/server.log
