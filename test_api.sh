#\!/bin/bash

echo "🧪 Testing Kimi-K2 API..."
echo "========================"

# Test health check
echo "1. Health check:"
curl -s http://localhost:10002/health || echo "Server not responding"

# Test chat completion
echo -e "\n2. Chat completion test:"
curl -X POST http://localhost:10002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, can you introduce yourself?"}
    ],
    "model": "kimi-k2",
    "temperature": 0.7,
    "max_tokens": 100,
    "stream": false
  }' | python3 -m json.tool

# Test streaming
echo -e "\n3. Streaming test:"
curl -X POST http://localhost:10002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Count from 1 to 5"}
    ],
    "model": "kimi-k2",
    "temperature": 0.7,
    "max_tokens": 50,
    "stream": true
  }'
