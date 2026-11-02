@echo off
title Setup Environment - Robust Plant Disease MobileNetV2
echo ====================================================================
echo Setting up Environment & Synthetic Reference Dataset
echo ====================================================================

if exist "C:\Users\itsuk\AppData\Local\Programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=C:\Users\itsuk\AppData\Local\Programs\Python\Python311\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [1/2] Installing / verifying dependencies...
%PYTHON_EXE% -m pip install -r requirements.txt

echo [2/2] Generating initial sample dataset for immediate testing...
%PYTHON_EXE% scripts/download_dataset.py --generate_synthetic --samples_per_class 15

echo ====================================================================
echo Environment setup completed!
echo To launch the application, run: run_demo.bat
echo ====================================================================
pause
