@echo off
REM Run the bot without the browser panel (credentials from .env). Useful on a VPS or
REM with Windows Task Scheduler ("At log on"). Create a file named STOP in this folder to
REM pause new entries; close the window or press Ctrl+C to stop.
cd /d "%~dp0\.."
".venv\Scripts\python.exe" -m tulipai live %*
pause
