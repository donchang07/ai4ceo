@echo off
REM ============================================================
REM runst.bat
REM 용도: uv/PATH 문제와 무관하게 Streamlit 실행
REM 동작: .venv 없으면 setup.bat 실행 후, .venv python으로 실행
REM 사용: runst <app.py> [streamlit options]
REM 예시: runst time.py --server.port 8502
REM ============================================================
setlocal
cd /d "%~dp0"

if "%~1"=="" (
  echo Usage: runst ^<app.py^> [streamlit options]
  echo Example: runst time.py --server.port 8502
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] .venv not found. Running setup first...
  call ".\setup.bat"
  if errorlevel 1 exit /b %ERRORLEVEL%
)

".\.venv\Scripts\python.exe" -m streamlit run %*
exit /b %ERRORLEVEL%
