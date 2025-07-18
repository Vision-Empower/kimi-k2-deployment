#!/usr/bin/env python3
"""
Test client for Kimi-K2 CPU+AMX server
"""

import requests
import json
import time
import argparse

def test_inference(host: str = "localhost", port: int = 8000):
    """Test inference endpoint"""
    url = f"http://{host}:{port}/v1/completions"
    
    # Test prompts
    prompts = [
        "What is the meaning of life?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about artificial intelligence.",
        "What are the benefits of renewable energy?",
    ]
    
    print(f"Testing Kimi-K2 server at {url}\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Test {i}: {prompt}")
        
        payload = {
            "model": "kimi-k2-instruct",
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            completion = result['choices'][0]['text']
            
            elapsed = time.time() - start_time
            tokens = len(completion.split())
            
            print(f"Response: {completion[:200]}...")
            print(f"Time: {elapsed:.2f}s | Tokens: ~{tokens} | Speed: ~{tokens/elapsed:.1f} tokens/s")
            print("-" * 80)
            
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            print("-" * 80)
        
        time.sleep(1)  # Brief pause between requests

def test_health(host: str = "localhost", port: int = 8000):
    """Test health endpoint"""
    url = f"http://{host}:{port}/health"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        print(f"✓ Server is healthy: {response.json()}")
    except:
        print("✗ Server health check failed")

def main():
    parser = argparse.ArgumentParser(description="Test Kimi-K2 CPU+AMX server")
    parser.add_argument('--host', type=str, default='localhost',
                      help='Server host')
    parser.add_argument('--port', type=int, default=8000,
                      help='Server port')
    parser.add_argument('--health-only', action='store_true',
                      help='Only check health')
    
    args = parser.parse_args()
    
    if args.health_only:
        test_health(args.host, args.port)
    else:
        test_health(args.host, args.port)
        print()
        test_inference(args.host, args.port)

if __name__ == "__main__":
    main()