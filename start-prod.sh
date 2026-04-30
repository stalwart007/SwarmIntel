#!/bin/bash

# SwarmIntel Production Start Script
echo "🚀 Starting SwarmIntel in Production Mode..."
echo "✨ Launching SwarmIntel on Port $PORT..."

# Use concurrently to run the Python backend and the Express proxy
# Frontend is already built in the Docker stage
npx concurrently \
  -n "backend,proxy" \
  -c "green,cyan" \
  "cd backend && uv run python run.py" \
  "node proxy.js"
