@echo off
REM One-shot research run on your broker's real gold history (MT5 must be installed and
REM logged in): download 2 years of M15 bars, backtest with benchmarks, walk-forward, ML.
cd /d "%~dp0\.."
set PY=".venv\Scripts\python.exe"
%PY% -m tulipai fetch --source mt5 --start 2024-09-01 --out data\xauusd_m15.csv || goto :fail
%PY% -m tulipai backtest --data data\xauusd_m15.csv --mc 200 --report reports\backtest.html || goto :fail
%PY% -m tulipai walkforward --data data\xauusd_m15.csv --report reports\walkforward.html || goto :fail
%PY% -m tulipai train-ml --data data\xauusd_m15.csv
echo.
echo Done. Open the reports folder: backtest.html and walkforward.html
start "" reports\walkforward.html
pause
goto :eof
:fail
echo Something failed - see the message above.
pause
