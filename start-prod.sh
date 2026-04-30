#!/bin/bash

# MiroFish / SwarmIntel Production Start Script

echo "🚀 Starting SwarmIntel in Production Mode..."

# 1. Build Frontend
echo "🏗️ Building Frontend..."
cd frontend
export VITE_API_BASE_URL="/api"
npm run build
cd ..

# 2. Start both Backend and Proxy using concurrently
echo "✨ Launching SwarmIntel on Port $PORT..."
# Use concurrently to run the Python backend and the Express proxy
# Backend on 5001, Proxy on PORT (default 3000)
npx concurrently \
  -n "backend,proxy" \
  -c "green,cyan" \
  "cd backend && uv run python run.py" \
  "node proxy.js"
