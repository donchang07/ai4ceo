@echo off
REM Same as setup.bat: setup.ps1 with automatic self-healing.
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 exit /b %ERRORLEVEL%
if exist "%~dp0.venv\Scripts\activate.bat" call "%~dp0.venv\Scripts\activate.bat"
exit /b %ERRORLEVEL%
