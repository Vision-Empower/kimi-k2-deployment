# Kimi-K2 CPU+AMX Deployment Strategy

## Overview

This document details our implementation of CPU-only inference for the Kimi-K2 1T parameter MoE model using Intel AMX (Advanced Matrix Extensions) optimization. This approach was developed after discovering that CloudExe provides remote GPU access rather than local GPU, making traditional CPU-GPU hybrid inference impractical due to network latency.

## Key Insights and Pivot

### Initial Approach (CPU-GPU Hybrid)
- **Goal**: Leverage both 2TB CPU memory and H100 GPU for mixed inference
- **Challenge**: CloudExe is a remote GPU execution service, not local GPU
- **Issue**: Network latency (even at 1Gbps) makes CPU-GPU data transfers prohibitive for real-time inference

### Revised Approach (CPU+AMX)
- **Solution**: Pure CPU inference with Intel AMX acceleration
- **Hardware**: Intel Xeon Platinum 8468 (96 cores/socket, dual socket = 192 cores)
- **Memory**: 2TB RAM with 812.8GB Kimi-K2 model pre-loaded in RAM disk
- **Optimization**: NUMA-aware expert placement and AMX INT8 quantization

## Technical Implementation

### 1. System Architecture
```
┌─────────────────────────────────────────┐
│         Kimi-K2 CPU Inference           │
├─────────────────────────────────────────┤
│   KTransformers + AMX Backend           │
├─────────────────────────────────────────┤
│      NUMA-Aware Expert Placement        │
├─────────────────────────────────────────┤
│  Intel Xeon 8468 (192 cores) + 2TB RAM │
└─────────────────────────────────────────┘
```

### 2. Key Components

#### KTransformers AMX Configuration (`kimi_k2_cpu_amx_config.yaml`)
- Configures all MoE experts to use CPU with AMX backend
- Enables INT8 quantization for memory efficiency
- Sets up module-based batching for expert parallelism

#### NUMA Optimizer (`numa_optimizer.py`)
- Detects dual-socket NUMA topology
- Distributes 128 experts across NUMA nodes
- Binds threads to local memory for reduced latency
- Monitors memory bandwidth and CPU utilization

#### CPU Inference Script (`cpu_amx_inference.py`)
- Provides both benchmarking and interactive modes
- Automatically detects AMX support
- Integrates NUMA optimization
- Handles graceful fallback to standard CPU ops

#### Server Deployment (`start_kimi_k2_server.sh`)
- Configures environment for optimal CPU performance
- Sets thread affinity and memory policies
- Launches KTransformers server with AMX config

### 3. Performance Optimizations

#### Memory Optimization
- **Model Loading**: Direct from RAM disk (/tmp/kimi-ramdisk)
- **Expert Caching**: Hot experts kept in L3 cache
- **NUMA Affinity**: Experts distributed based on access patterns

#### Compute Optimization
- **AMX Utilization**: INT8 operations for 4x throughput
- **Thread Pooling**: 192 threads with NUMA awareness
- **Batch Processing**: Module-based batching for expert parallelism

#### Expected Performance
- **Prefill**: ~5-10 tokens/second (CPU bottleneck)
- **Generation**: ~2-4 tokens/second (memory bandwidth limited)
- **Batch Size**: 1-4 for optimal throughput

## Deployment Guide

### Prerequisites
1. Intel Xeon Scalable processor with AMX support
2. 2TB RAM minimum
3. Ubuntu 22.04 or later
4. KTransformers installed

### Setup Steps

1. **Verify AMX Support**:
```bash
lscpu | grep amx
# Should show: amx_bf16, amx_tile, amx_int8
```

2. **Configure NUMA**:
```bash
numactl --hardware
# Verify 2 nodes with balanced memory
```

3. **Start Server**:
```bash
cd /root
./start_kimi_k2_server.sh
```

4. **Test Inference**:
```bash
./test_kimi_k2_client.py
```

## Monitoring and Debugging

### Performance Monitoring
```bash
# CPU and memory usage
htop

# NUMA statistics  
numastat -n

# AMX utilization
perf stat -e cpu/event=0xc7,umask=0x10/ ./cpu_amx_inference.py
```

### Common Issues

1. **Low Performance**:
   - Check AMX detection in logs
   - Verify NUMA binding is active
   - Monitor memory bandwidth saturation

2. **High Latency**:
   - Reduce batch size
   - Check for CPU throttling
   - Verify expert cache hit rate

3. **Memory Errors**:
   - Ensure 2TB RAM available
   - Check NUMA memory distribution
   - Verify model loaded in RAM disk

## Future Optimizations

1. **Dynamic Expert Selection**: Predict and preload frequently used experts
2. **Adaptive Batching**: Adjust batch size based on workload
3. **Kernel Fusion**: Combine operations to reduce memory transfers
4. **Multi-Instance**: Run multiple model instances across NUMA nodes

## Conclusion

While the initial plan was CPU-GPU hybrid inference, the discovery that CloudExe provides remote GPU access led to a pivot toward CPU-only inference with AMX optimization. This approach effectively utilizes the available 2TB RAM and 192-core CPU to run the 1T parameter Kimi-K2 model, albeit at lower throughput than GPU inference would provide.

The key insight: "再仔细想想，结合具体情况具体分析" (think carefully, analyze the specific situation) - led to a practical solution that works within the actual constraints rather than theoretical ideals.