@echo off
cd /d "%~dp0"
if not exist "%APPDATA%\open-webui\python\pythonw.exe" (
 echo Python runtime not found. See README.md.
 pause
 exit /b 1
)
start "" "%APPDATA%\open-webui\python\pythonw.exe" "%~dp0launcher.py"
