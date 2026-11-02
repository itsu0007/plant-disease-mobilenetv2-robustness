@echo off
title Plant Disease MobileNetV2 Robustness - Web App
echo ====================================================================
echo Starting Robust Plant Disease Detection Web App (Streamlit)...
echo ====================================================================

REM Locate Python
if exist "C:\Users\itsuk\AppData\Local\Programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=C:\Users\itsuk\AppData\Local\Programs\Python\Python311\python.exe"
) else (
    set "PYTHON_EXE=python"
)

REM Run Streamlit
%PYTHON_EXE% -m streamlit run dashboard.py --server.port 8501 --browser.gatherUsageStats false
pause
