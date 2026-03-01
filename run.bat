@echo off
REM CrystalCare Run Script
REM Runs CrystalCare from the cc conda environment

cd /d "%~dp0"

set CONDA_PYTHON=%USERPROFILE%\.conda\envs\cc\python.exe

if not exist "%CONDA_PYTHON%" (
    echo [ERROR] Python not found at: %CONDA_PYTHON%
    echo.
    echo Please ensure the 'cc' conda environment exists with Python 3.10
    pause
    exit /b 1
)

echo [OK] Using Python: %CONDA_PYTHON%
echo Starting CrystalCare...
echo.

"%CONDA_PYTHON%" main.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] CrystalCare exited with an error
    pause
)
