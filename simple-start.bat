@echo off
REM Simple Startup Script for Windows

echo 🚀 Starting Biometric Text Analysis System...
echo ==================================================

cd /d "%~dp0"
set PROJECT_ROOT=%CD%

echo 🐳 Starting Docker containers...
cd infra
docker-compose up -d db redis
echo ✅ Database services started

echo.
echo 🔧 Starting Backend...
cd /d "%PROJECT_ROOT%\backend"
start "Backend API" ..\venv\Scripts\uvicorn.exe app:app --reload --port 8000
echo ⏳ Backend starting... (wait 10 seconds)

echo.
echo 🎨 Starting Frontend...
cd /d "%PROJECT_ROOT%\frontend\widget"
start "Frontend" npm run dev
echo ⏳ Frontend starting... (wait 5 seconds)

echo.
echo 🎉 SUCCESS! Services are starting...
echo ==================================================
echo 🗄️  PostgreSQL:  localhost:5432
echo 📱 Redis:        localhost:6379
echo 🔧 Backend API:  http://localhost:8000
echo 🎨 Frontend:     http://localhost:3000
echo.
echo ⏳ Please wait 15 seconds for everything to be ready
echo 🌐 Then open: http://localhost:3000
echo.
echo 🛑 To stop: run stop-all.bat
echo.
pause