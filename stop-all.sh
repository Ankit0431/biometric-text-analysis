#!/bin/bash
# Biometric Text Analysis - Stop All Services Script

echo "🛑 Stopping Biometric Text Analysis System..."
echo "=============================================="

# Change to project root
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)

# Stop backend and frontend processes
if [ -f ".service_pids" ]; then
    source .service_pids
    echo "🔧 Stopping backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null || true

    echo "🎨 Stopping frontend (PID: $FRONTEND_PID)..."
    kill $FRONTEND_PID 2>/dev/null || true

    rm .service_pids
else
    echo "🔄 Killing any uvicorn and vite processes..."
    pkill -f "uvicorn app:app" || true
    pkill -f "vite" || true
fi

# Stop Docker containers
echo "🐳 Stopping Docker containers..."
cd "$PROJECT_ROOT/infra"
docker-compose down

echo "✅ All services stopped successfully!"