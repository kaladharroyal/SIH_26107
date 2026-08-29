@echo off
title BIS AI Assistant - Live Server
cd /d "%~dp0"

echo ================================================================
echo          BUREAU OF INDIAN STANDARDS - AI COMPLIANCE SYSTEM
echo ================================================================
echo.
echo Starting BIS AI Assistant Server...
echo Portal URL: http://127.0.0.1:8000
echo.

if exist ".venv\Scripts\python.exe" (
    start "" http://127.0.0.1:8000
    ".venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
) else if exist "..\.venv\Scripts\python.exe" (
    start "" http://127.0.0.1:8000
    "..\.venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
) else (
    start "" http://127.0.0.1:8000
    python -m uvicorn app:app --host 127.0.0.1 --port 8000
)

pause
