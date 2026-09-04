@echo off
setlocal
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo Python environment not found.
    echo Run setup_wan_4050.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

rem RTX 4050 6 GB / 16 GB RAM quality-safe defaults
set WAN_BACKEND=gguf
set WAN_OFFLOAD_MODE=sequential
set WAN_DELIVERY_FPS=32
set WAN_SMOOTHING=auto
set WAN_UPSCALER=auto
set REALESRGAN_TILE=256
set LTX_MAX_GPU_WORKERS=1

echo ============================================================
echo  Wan2.1 GGUF - RTX 4050 6 GB profile
echo ============================================================
python setup_wan.py --status
echo.
echo Starting local video studio...
python run.py --video-studio-ui

if errorlevel 1 (
    echo.
    echo The video studio stopped with an error.
    pause
)
