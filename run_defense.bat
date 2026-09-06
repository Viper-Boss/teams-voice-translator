@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Please run setup.ps1 first.
  pause
  exit /b 1
)
start "Defense Mode" ".venv\Scripts\pythonw.exe" defense_main.py
