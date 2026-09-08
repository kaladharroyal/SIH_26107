@echo off
title BIS AI Assistant - Ultra Fast Mode
cd /d "%~dp0"

echo ================================================================
echo     BUREAU OF INDIAN STANDARDS - AI COMPLIANCE SYSTEM (FAST MODE)
echo ================================================================
echo.
echo Starting in Ultra-Fast Mode (BM25 Sparse Search + Sub-Second Startup)...
echo Portal URL: http://127.0.0.1:8000
echo.

set USE_FAST_RETRIEVAL=true

rem Launch browser automatically as soon as port 8000 is listening
start "" /b powershell -NoProfile -Command "while ((Test-NetConnection 127.0.0.1 -Port 8000 -WarningAction SilentlyContinue).TcpTestSucceeded -ne $true) { Start-Sleep -Milliseconds 250 }; Start-Process 'http://127.0.0.1:8000'"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
) else if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
) else (
    python -m uvicorn app:app --host 127.0.0.1 --port 8000
)

pause
