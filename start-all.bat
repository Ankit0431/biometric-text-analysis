@echo off
REM Biometric Text Analysis - Complete Startup Script (Windows)
REM This script starts all required services in the correct order

echo 🚀 Starting Biometric Text Analysis System...
echo ==================================================

REM Change to project root directory
cd /d "%~dp0"
set PROJECT_ROOT=%CD%

echo 📁 Project root: %PROJECT_ROOT%

REM Step 1: Start Docker Desktop if not running
echo.
echo 🐳 Step 1: Checking Docker...
docker ps >nul 2>&1
if errorlevel 1 (
    echo 📦 Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

    echo ⏳ Waiting for Docker to start...
    :docker_wait
    timeout /t 5 /nobreak >nul
    docker ps >nul 2>&1
    if errorlevel 1 goto docker_wait
)
echo ✅ Docker is running

REM Step 2: Start PostgreSQL and Redis
echo.
echo 🗄️ Step 2: Starting Database Services...
cd /d "%PROJECT_ROOT%\infra"

REM Stop any existing containers
echo 🛑 Stopping existing containers...
docker-compose down

REM Start database services
echo 🚀 Starting PostgreSQL and Redis...
docker-compose up -d db redis

REM Wait for database to be ready
echo ⏳ Waiting for PostgreSQL to be ready...
:db_wait
timeout /t 2 /nobreak >nul
docker-compose exec -T db pg_isready -U postgres >nul 2>&1
if errorlevel 1 goto db_wait
echo ✅ PostgreSQL is ready

echo ✅ Database services started successfully

REM Step 3: Start Backend
echo.
echo 🔧 Step 3: Starting Backend API...
cd /d "%PROJECT_ROOT%\backend"

REM Check if virtual environment exists
if not exist "..\venv\" (
    echo ❌ Virtual environment not found at ..\venv
    echo Please run: python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
    pause
    exit /b 1
)

REM Kill any existing backend process
echo 🛑 Stopping existing backend...
taskkill /f /im "uvicorn.exe" >nul 2>&1

REM Start backend in background
echo 🚀 Starting backend server...
start "Backend API" /min ..\venv\Scripts\uvicorn.exe app:app --reload --port 8000

REM Wait for backend to start (simple timeout instead of health check)
echo ⏳ Waiting for backend to start...
timeout /t 8 /nobreak >nul
echo ✅ Backend should be ready now

REM Step 4: Start Frontend
echo.
echo 🎨 Step 4: Starting Frontend...
cd /d "%PROJECT_ROOT%\frontend\widget"

REM Install dependencies if node_modules doesn't exist
if not exist "node_modules\" (
    echo 📦 Installing npm dependencies...
    npm install
)

REM Kill any existing frontend process
echo 🛑 Stopping existing frontend...
taskkill /f /im "node.exe" >nul 2>&1

REM Start frontend in background
echo 🚀 Starting frontend server...
start "Frontend" /min npm run dev

REM Wait for frontend to start
echo ⏳ Waiting for frontend to start...
timeout /t 5 /nobreak >nul
echo ✅ Frontend should be ready now

set FRONTEND_PORT=3000

REM Step 5: Success Summary
echo.
echo 🎉 SUCCESS! All services are running:
echo ==================================================
echo 🗄️  PostgreSQL:  localhost:5432
echo 📱 Redis:        localhost:6379
echo 🔧 Backend API:  http://localhost:8000
echo 🎨 Frontend:     http://localhost:%FRONTEND_PORT%
echo.
echo 📖 API Documentation: http://localhost:8000/docs
echo 🧪 Health Check:      http://localhost:8000/health
echo.
echo 🌐 Opening browser to: http://localhost:%FRONTEND_PORT%
echo.

REM Open browser
start http://localhost:%FRONTEND_PORT%

echo 🛑 To stop all services, run: stop-all.bat
echo Or close this window and run: docker-compose -f infra\docker-compose.yml down
echo.
echo 📊 Press any key to stop all services...
pause >nul

:cleanup
echo.
echo 🛑 Shutting down services...
taskkill /f /im "uvicorn.exe" >nul 2>&1
taskkill /f /im "node.exe" >nul 2>&1
cd /d "%PROJECT_ROOT%\infra"
docker-compose down
echo ✅ All services stopped
pause