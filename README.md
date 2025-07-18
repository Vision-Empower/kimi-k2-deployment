# Kimi-K2 Deployment Guide

This repository documents the deployment process of Kimi-K2 (1T parameter MoE model) using KTransformers on a high-performance server.

## 🆕 CPU+AMX Deployment Strategy

We've developed a CPU-only inference approach using Intel AMX (Advanced Matrix Extensions) after discovering that CloudExe provides remote GPU access rather than local GPU. This makes traditional CPU-GPU hybrid inference impractical due to network latency. See [KIMI_K2_CPU_AMX_DEPLOYMENT.md](KIMI_K2_CPU_AMX_DEPLOYMENT.md) for details.

## 🚀 Quick Start

```bash
# Clone this repository
git clone https://github.com/yourusername/kimi-k2-deployment.git
cd kimi-k2-deployment

# Run the deployment script
./deploy.sh
```

## 📋 System Requirements

### Hardware
- **CPU**: Intel Xeon Platinum 8468 (2x48 cores, 192 threads)
- **Memory**: 2TB RAM (minimum 600GB for Q4_K_M model)
- **GPU**: NVIDIA H100 80GB (or similar, 14GB+ VRAM required)
- **Storage**: 1TB+ for model files

### Software
- Ubuntu 22.04 LTS
- CUDA 12.2
- Python 3.10+
- PyTorch 2.3.0+cu121

## 🛠️ Installation Steps

### 1. Environment Setup

```bash
# Install system dependencies
apt-get update
apt-get install -y git cmake ninja-build

# Install Python dependencies
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate einops
```

### 2. Download Models

```bash
# Download original model
huggingface-cli download moonshotai/Kimi-K2-Instruct --local-dir ./kimi-k2-instruct

# Download GGUF quantized model
huggingface-cli download KVCache-ai/Kimi-K2-Instruct-GGUF --local-dir ./kimi-k2-gguf
```

### 3. Install KTransformers

```bash
# Clone and install KTransformers
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
export CUDA_HOME=/usr/local/cuda-12.2
export TORCH_CUDA_ARCH_LIST="9.0"  # For H100
pip install -e . --no-deps --no-build-isolation
```

### 4. Run the Server

```bash
# Start with conservative settings
python -m ktransformers.server.main \
  --port 10002 \
  --model_path ./kimi-k2-instruct \
  --gguf_path ./kimi-k2-gguf \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml \
  --max_new_tokens 512 \
  --cache_lens 4096 \
  --chunk_size 64 \
  --max_batch_size 1 \
  --backend_type ktransformers
```

## 🐛 Common Issues and Solutions

### Issue 1: CUDA Version Mismatch
**Error**: `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED`

**Solution**: Ensure PyTorch CUDA version matches system CUDA

### Issue 2: GPU Memory Limitations (MIG Mode)
**Error**: CUDA out of memory with MIG enabled

**Solution**: 
- Option 1: Disable MIG mode (requires admin)
- Option 2: Use CPU-only mode
- Option 3: Use smaller quantization

### Issue 3: Compilation Errors
**Error**: Missing symbols or ABI incompatibility

**Solution**: Clean rebuild with correct flags

## 🔧 CPU+AMX Deployment (NEW)

For systems with Intel AMX support and large RAM capacity:

```bash
# Check AMX support
lscpu | grep amx

# Start CPU+AMX optimized server
cd scripts
./start_kimi_k2_server.sh

# Test inference
python test_kimi_k2_client.py
```

### CPU+AMX Configuration
- Uses Intel AMX for INT8 acceleration
- NUMA-aware expert distribution
- Optimized for 192-core Intel Xeon systems
- See `configs/kimi_k2_cpu_amx_config.yaml` for details

## 📊 Performance Metrics

| Configuration | Memory Usage | GPU VRAM | Speed (tokens/s) |
|--------------|--------------|----------|------------------|
| Q4_K_M + H100 | ~600GB | 14GB | 10-14 |
| Q4_K_M + CPU only | ~600GB | 0 | 8-10 |
| Q2_K + H100 | ~300GB | 10GB | 12-16 |
| INT8 + AMX (CPU) | ~812GB | 0 | 2-4 |

## 🤝 Contributing

Please feel free to submit issues and pull requests\!

## 📄 License

MIT License
