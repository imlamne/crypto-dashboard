@echo off
echo Starting Crypto Dashboard...
echo.
echo [1/2] Starting data scheduler (fetches every 15 minutes)...
start "Crypto Scheduler" cmd /k "cd /d %~dp0 && python main.py"
echo [2/2] Starting dashboard...
timeout /t 3 /nobreak > nul
start "Crypto Dashboard" cmd /k "cd /d %~dp0 && streamlit run src/dashboard.py"
echo.
echo Both services started! Dashboard will open at http://localhost:8501
