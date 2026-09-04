@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  LTX Video - Wan2.1 GGUF setup for RTX 4050 6 GB
echo ============================================================

if not exist venv\Scripts\python.exe (
    echo Creating Python virtual environment...
    py -3 -m venv venv
    if errorlevel 1 goto :error
)

call venv\Scripts\activate.bat
if errorlevel 1 goto :error

echo Installing/updating Python requirements...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Downloading Wan2.1 Q5_0 GGUF low-memory setup...
python setup_wan.py --download-gguf
if errorlevel 1 goto :error

echo.
echo Setup complete.
echo Run run_wan_4050.bat to start the video studio.
pause
exit /b 0

:error
echo.
echo Setup failed. Review the message above and try again.
pause
exit /b 1
