#!/bin/bash
# Biometric Text Analysis - Complete Startup Script
# This script starts all required services in the correct order

set -e  # Exit on any error

echo "🚀 Starting Biometric Text Analysis System..."
echo "=================================================="

# Change to project root directory
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)

echo "📁 Project root: $PROJECT_ROOT"

# Function to check if a port is in use
check_port() {
    if netstat -an | grep -q ":$1 "; then
        echo "⚠️  Port $1 is already in use"
        return 0
    else
        return 1
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1

    echo "⏳ Waiting for $service_name to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        echo "   Attempt $attempt/$max_attempts - waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo "❌ $service_name failed to start after $max_attempts attempts"
    return 1
}

# Step 1: Start Docker Desktop if not running
echo ""
echo "🐳 Step 1: Checking Docker..."
if ! docker ps > /dev/null 2>&1; then
    echo "📦 Starting Docker Desktop..."
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        # Windows
        "/c/Program Files/Docker/Docker/Docker Desktop.exe" &
    else
        # Linux/Mac
        open -a Docker || systemctl start docker
    fi

    echo "⏳ Waiting for Docker to start..."
    while ! docker ps > /dev/null 2>&1; do
        sleep 5
        echo "   Still waiting for Docker..."
    done
fi
echo "✅ Docker is running"

# Step 2: Start PostgreSQL and Redis
echo ""
echo "🗄️ Step 2: Starting Database Services..."
cd "$PROJECT_ROOT/infra"

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Start database services
echo "🚀 Starting PostgreSQL and Redis..."
docker-compose up -d db redis

# Wait for database to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
while ! docker-compose exec -T db pg_isready -U postgres > /dev/null 2>&1; do
    sleep 2
    echo "   Waiting for PostgreSQL..."
done
echo "✅ PostgreSQL is ready"

echo "✅ Database services started successfully"

# Step 3: Start Backend
echo ""
echo "🔧 Step 3: Starting Backend API..."
cd "$PROJECT_ROOT/backend"

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "❌ Virtual environment not found at ../venv"
    echo "Please run: python -m venv venv && source venv/Scripts/activate && pip install -r requirements.txt"
    exit 1
fi

# Kill any existing backend process
echo "🛑 Stopping existing backend..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    taskkill //f //im "uvicorn.exe" 2>/dev/null || true
else
    # Linux/Mac
    pkill -f "uvicorn app:app" || true
fi

# Start backend in background
echo "🚀 Starting backend server..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    ../venv/Scripts/uvicorn.exe app:app --reload --port 8000 &
else
    # Linux/Mac
    ../venv/bin/uvicorn app:app --reload --port 8000 &
fi

BACKEND_PID=$!
echo "📝 Backend PID: $BACKEND_PID"

# Wait for backend to be ready
if wait_for_service "http://localhost:8000/health" "Backend API"; then
    echo "✅ Backend started successfully"
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID || true
    exit 1
fi

# Step 4: Start Frontend
echo ""
echo "🎨 Step 4: Starting Frontend..."
cd "$PROJECT_ROOT/frontend/widget"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing npm dependencies..."
    npm install
fi

# Kill any existing frontend process
echo "🛑 Stopping existing frontend..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    taskkill //f //im "node.exe" 2>/dev/null || true
else
    # Linux/Mac
    pkill -f "vite" || true
fi

# Start frontend in background
echo "🚀 Starting frontend server..."
npm run dev &
FRONTEND_PID=$!
echo "📝 Frontend PID: $FRONTEND_PID"

# Wait a bit for frontend to start
sleep 5

# Try different ports for frontend
FRONTEND_PORT=""
for port in 3000 3001 3002 3003; do
    if wait_for_service "http://localhost:$port" "Frontend (port $port)"; then
        FRONTEND_PORT=$port
        break
    fi
done

if [ -z "$FRONTEND_PORT" ]; then
    echo "❌ Frontend failed to start on any port"
    kill $BACKEND_PID $FRONTEND_PID || true
    exit 1
fi

# Step 5: Success Summary
echo ""
echo "🎉 SUCCESS! All services are running:"
echo "=================================================="
echo "🗄️  PostgreSQL:  localhost:5432"
echo "📱 Redis:        localhost:6379"
echo "🔧 Backend API:  http://localhost:8000"
echo "🎨 Frontend:     http://localhost:$FRONTEND_PORT"
echo ""
echo "📖 API Documentation: http://localhost:8000/docs"
echo "🧪 Health Check:      http://localhost:8000/health"
echo ""
echo "🌐 Open your browser to: http://localhost:$FRONTEND_PORT"
echo ""
echo "🛑 To stop all services, run: ./stop-all.sh"
echo "Or press Ctrl+C and run: docker-compose -f infra/docker-compose.yml down"

# Save PIDs for stop script
echo "BACKEND_PID=$BACKEND_PID" > "$PROJECT_ROOT/.service_pids"
echo "FRONTEND_PID=$FRONTEND_PID" >> "$PROJECT_ROOT/.service_pids"
echo "FRONTEND_PORT=$FRONTEND_PORT" >> "$PROJECT_ROOT/.service_pids"

# Keep script running and show logs
echo ""
echo "📊 Monitoring services (Press Ctrl+C to stop)..."
echo "=================================================="

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    cd "$PROJECT_ROOT/infra"
    docker-compose down
    echo "✅ All services stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Monitor services
while true; do
    sleep 5

    # Check if processes are still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "❌ Backend process died"
        cleanup
    fi

    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "❌ Frontend process died"
        cleanup
    fi

    # Show brief status
    echo "$(date '+%H:%M:%S') - ✅ All services running (Backend: $BACKEND_PID, Frontend: $FRONTEND_PID)"
done