@echo off
REM One-shot research on your broker's real gold history. MT5 must be installed (and
REM logged in, or credentials in .env). Downloads ~2 years of M15 bars, then runs the
REM walk-forward test + benchmarks, a frozen-settings check and the ML filter; writes:
REM   reports\research_summary.txt   <- send this file to Claude
REM   reports\backtest.html, reports\walkforward.html
REM Takes roughly 5-30 minutes depending on your PC.
setlocal
cd /d "%~dp0\.."

where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3) else (set PY=python)
if not exist ".venv\Scripts\python.exe" (
  echo [TulipAI] First run: creating a Python environment...
  %PY% -m venv .venv || goto :fail
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
  ".venv\Scripts\python.exe" -m pip install -e . || goto :fail
)

".venv\Scripts\python.exe" -m tulipai research %* || goto :fail
echo.
echo Done. Send reports\research_summary.txt to Claude.
start "" reports
start "" reports\walkforward.html
pause
goto :eof

:fail
echo.
echo Something failed - see the message above (copy it to Claude if unsure).
pause
