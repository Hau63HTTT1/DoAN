@echo off
setlocal enabledelayedexpansion

:: Get arguments
set IMAGE_NAME=%1
set RESOLUTION=%2
if "%RESOLUTION%"=="" set RESOLUTION=256

:: Set CUDA environment variables
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

:: Change to PiFuHD directory
cd /d "%~dp0..\model\pifuhd"

echo Running PiFuHD...
python -m apps.simple_test -i "sample_images" -o "results" -c "checkpoints/pifuhd.pt" -r %RESOLUTION% --use_rect

echo Cleaning generated mesh...
python apps/clean_mesh.py -f "results/pifuhd_final/recon"

echo Process completed!
exit /b 0 