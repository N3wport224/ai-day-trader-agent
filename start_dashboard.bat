@echo off
REM AI Day Trader - double-click to start the dashboard on Windows.
REM First run creates a private Python environment in .venv and installs packages.
setlocal
cd /d "%~dp0"
title AI Day Trader

where py >nul 2>nul
if errorlevel 1 goto :nopython

if exist ".venv\Scripts\python.exe" goto :run

echo First run: setting up. This takes a few minutes...
py -3.12 -m venv .venv
if errorlevel 1 py -3 -m venv .venv
if errorlevel 1 goto :setupfailed
".venv\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info < (3, 10))"
if errorlevel 1 goto :oldpython
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :setupfailed

:run
echo Starting the dashboard. Keep this window open while you trade; close it to stop the dashboard.
".venv\Scripts\python.exe" start_dashboard.py
pause
exit /b 0

:nopython
echo Python is not installed. Install Python 3.12 from https://www.python.org/downloads/
echo During setup, tick "Add python.exe to PATH". Then double-click this file again.
pause
exit /b 1

:oldpython
echo Your Python is older than 3.10. Install Python 3.12 from https://www.python.org/downloads/
rmdir /s /q .venv
pause
exit /b 1

:setupfailed
echo Setup failed - see the messages above. Delete the .venv folder and try again.
pause
exit /b 1
