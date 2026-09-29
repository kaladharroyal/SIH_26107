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

set "PY_CMD="

if exist ".venv\Scripts\uvicorn.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
) else if exist "..\.venv\Scripts\uvicorn.exe" (
    set "PY_CMD=..\.venv\Scripts\python.exe"
) else if exist "venv\Scripts\uvicorn.exe" (
    set "PY_CMD=venv\Scripts\python.exe"
) else if exist "..\venv\Scripts\uvicorn.exe" (
    set "PY_CMD=..\venv\Scripts\python.exe"
) else (
    set "PY_CMD=python"
)

start "" http://127.0.0.1:8000
%PY_CMD% -m uvicorn app:app --host 127.0.0.1 --port 8000

pause
