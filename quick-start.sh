#!/bin/bash
# Simple Startup Script - Just starts everything and exits

echo "🚀 Starting Biometric Text Analysis System..."

# Change to project root
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)

# Start Docker containers
echo "🐳 Starting Docker containers..."
cd "$PROJECT_ROOT/infra"
docker-compose up -d db redis

# Wait a moment for DB to be ready
echo "⏳ Waiting for database..."
sleep 5

# Start Backend
echo "🔧 Starting Backend..."
cd "$PROJECT_ROOT/backend"
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    start "Backend API" ../venv/Scripts/uvicorn.exe app:app --reload --port 8000
else
    # Linux/Mac
    ../venv/bin/uvicorn app:app --reload --port 8000 &
fi

# Wait a moment for backend to start
echo "⏳ Waiting for backend..."
sleep 8

# Start Frontend
echo "🎨 Starting Frontend..."
cd "$PROJECT_ROOT/frontend/widget"
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    start "Frontend" npm run dev
else
    # Linux/Mac
    npm run dev &
fi

echo ""
echo "🎉 All services are starting!"
echo "================================"
echo "🗄️  PostgreSQL:  localhost:5432"
echo "📱 Redis:        localhost:6379"
echo "🔧 Backend API:  http://localhost:8000"
echo "🎨 Frontend:     http://localhost:3000 or 3001"
echo ""
echo "⏳ Please wait 10-15 seconds for everything to be ready"
echo "🌐 Then open: http://localhost:3000"
echo ""
echo "🛑 To stop: run ./stop-all.sh"