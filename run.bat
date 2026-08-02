@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Please run setup.ps1 first.
  pause
  exit /b 1
)
start "Teams Voice Translator" ".venv\Scripts\pythonw.exe" main.py
