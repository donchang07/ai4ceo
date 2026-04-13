@echo off
REM Thin wrapper: CMD resolves "setup" / ".\setup" to setup.bat before .cmd.
REM Runs setup.ps1 with automatic self-healing (no user options needed).
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 exit /b %ERRORLEVEL%
if exist "%~dp0.venv\Scripts\activate.bat" call "%~dp0.venv\Scripts\activate.bat"
exit /b %ERRORLEVEL%
