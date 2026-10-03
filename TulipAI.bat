@echo off
REM Double-click to start TulipAI on Windows: sets up Python packages on first run,
REM then opens the control panel in your browser (login, start/stop, dashboard).
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3) else (set PY=python)

if not exist ".venv\Scripts\python.exe" (
  echo [TulipAI] First run: creating a Python environment...
  %PY% -m venv .venv || goto :nopython
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
  ".venv\Scripts\python.exe" -m pip install -e . || goto :fail
)

echo [TulipAI] Opening the control panel at http://127.0.0.1:8765 ...
".venv\Scripts\python.exe" -m tulipai panel
goto :eof

:nopython
echo.
echo Python 3.10+ was not found. Install it from https://www.python.org/downloads/windows/
echo (tick "Add python.exe to PATH" during setup), then double-click this file again.
pause
goto :eof

:fail
echo.
echo Installing packages failed - check your internet connection and try again.
pause
