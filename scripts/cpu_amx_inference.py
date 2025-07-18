#!/usr/bin/env python3
"""
CPU+AMX inference script for Kimi-K2 model
Supports both benchmarking and interactive modes
"""

import os
import sys
import time
import torch
import argparse
import numpy as np
from typing import Optional, List, Dict
import logging

# Add ktransformers to path
sys.path.append('/root/ktransformers')

from ktransformers import KTransformersModel
from numa_optimizer import setup_numa_optimized_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KimiK2CPUInference:
    def __init__(self, model_path: str, use_amx: bool = True):
        self.model_path = model_path
        self.use_amx = use_amx
        self.model = None
        
        # Setup NUMA optimization
        self.numa_optimizer = setup_numa_optimized_inference()
        
        # Check AMX support
        if self.use_amx:
            self._check_amx_support()
    
    def _check_amx_support(self):
        """Check if CPU supports Intel AMX"""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            flags = info.get('flags', [])
            
            amx_flags = ['amx_bf16', 'amx_tile', 'amx_int8']
            has_amx = all(flag in flags for flag in amx_flags)
            
            if has_amx:
                logger.info("✓ Intel AMX support detected")
            else:
                logger.warning("✗ Intel AMX not supported on this CPU")
                self.use_amx = False
        except:
            logger.warning("Could not detect AMX support, proceeding without AMX")
            self.use_amx = False
    
    def load_model(self, config_path: str):
        """Load Kimi-K2 model with CPU+AMX configuration"""
        logger.info(f"Loading model from {self.model_path}")
        logger.info(f"Using config: {config_path}")
        
        # Set environment for optimal CPU performance
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['OMP_WAIT_POLICY'] = 'ACTIVE'
        
        start_time = time.time()
        
        try:
            self.model = KTransformersModel.from_config(
                config_path,
                gguf_path=self.model_path,
                device="cpu",
                dtype="int8" if self.use_amx else "float16"
            )
            
            load_time = time.time() - start_time
            logger.info(f"✓ Model loaded in {load_time:.2f} seconds")
            
            # Get model info
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model parameters: {param_count / 1e9:.1f}B")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def benchmark_inference(self, prompt: str, max_tokens: int = 100):
        """Benchmark inference performance"""
        logger.info("\n=== Benchmarking CPU+AMX Inference ===")
        
        # Warmup
        logger.info("Warming up...")
        _ = self.model.generate(prompt, max_length=10)
        
        # Benchmark prefill
        logger.info("\nBenchmarking prefill...")
        start_time = time.time()
        tokens = self.model.tokenize(prompt)
        prefill_time = time.time() - start_time
        
        num_tokens = len(tokens)
        prefill_speed = num_tokens / prefill_time
        logger.info(f"Prefill: {num_tokens} tokens in {prefill_time:.2f}s")
        logger.info(f"Speed: {prefill_speed:.1f} tokens/second")
        
        # Benchmark generation
        logger.info("\nBenchmarking generation...")
        start_time = time.time()
        output = self.model.generate(
            prompt,
            max_length=max_tokens,
            temperature=0.7,
            top_p=0.9
        )
        gen_time = time.time() - start_time
        
        # Calculate metrics
        generated_tokens = len(self.model.tokenize(output)) - num_tokens
        gen_speed = generated_tokens / gen_time
        
        logger.info(f"Generated: {generated_tokens} tokens in {gen_time:.2f}s")
        logger.info(f"Speed: {gen_speed:.1f} tokens/second")
        
        # Memory usage
        import psutil
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024**3)
        logger.info(f"\nMemory usage: {memory_gb:.1f} GB")
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"CPU usage: {cpu_percent:.1f}%")
        
        return {
            'prefill_speed': prefill_speed,
            'generation_speed': gen_speed,
            'memory_gb': memory_gb,
            'cpu_percent': cpu_percent
        }
    
    def interactive_inference(self):
        """Interactive inference mode"""
        logger.info("\n=== Interactive CPU+AMX Inference ===")
        logger.info("Type 'exit' to quit, 'bench' to run benchmark\n")
        
        while True:
            try:
                prompt = input("\nPrompt> ").strip()
                
                if prompt.lower() == 'exit':
                    break
                elif prompt.lower() == 'bench':
                    test_prompt = "Explain the concept of machine learning in simple terms."
                    self.benchmark_inference(test_prompt)
                    continue
                elif not prompt:
                    continue
                
                # Generate response
                logger.info("Generating response...")
                start_time = time.time()
                
                response = self.model.generate(
                    prompt,
                    max_length=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                gen_time = time.time() - start_time
                
                # Display response
                print(f"\nResponse: {response}")
                print(f"\nGeneration time: {gen_time:.2f}s")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Kimi-K2 CPU+AMX Inference")
    parser.add_argument('--model-path', type=str, 
                      default='/tmp/kimi-ramdisk/Kimi-K2-Instruct',
                      help='Path to Kimi-K2 model')
    parser.add_argument('--config', type=str,
                      default='/root/kimi_k2_cpu_amx_config.yaml',
                      help='Path to KTransformers config')
    parser.add_argument('--benchmark', action='store_true',
                      help='Run benchmark mode')
    parser.add_argument('--no-amx', action='store_true',
                      help='Disable AMX optimization')
    parser.add_argument('--prompt', type=str,
                      help='Single prompt for inference')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = KimiK2CPUInference(
        model_path=args.model_path,
        use_amx=not args.no_amx
    )
    
    # Load model
    engine.load_model(args.config)
    
    # Run inference
    if args.benchmark:
        test_prompt = args.prompt or "Explain quantum computing to a 10-year-old."
        engine.benchmark_inference(test_prompt)
    elif args.prompt:
        response = engine.model.generate(args.prompt, max_length=256)
        print(f"\nResponse: {response}")
    else:
        engine.interactive_inference()

if __name__ == "__main__":
    main()