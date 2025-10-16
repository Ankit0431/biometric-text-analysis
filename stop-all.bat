@echo off
REM Biometric Text Analysis - Stop All Services Script (Windows)

echo 🛑 Stopping Biometric Text Analysis System...
echo ==============================================

REM Change to project root
cd /d "%~dp0"

REM Stop processes
echo 🔧 Stopping backend processes...
taskkill /f /im "uvicorn.exe" >nul 2>&1

echo 🎨 Stopping frontend processes...
taskkill /f /im "node.exe" >nul 2>&1

REM Stop Docker containers
echo 🐳 Stopping Docker containers...
cd infra
docker-compose down

echo ✅ All services stopped successfully!
pause