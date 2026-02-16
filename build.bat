@echo off
REM CrystalCare Build Script
REM Compiles CrystalCare into a standalone executable using Nuitka

echo ============================================================
echo CrystalCare - Nuitka Compilation Script
echo ============================================================
echo.

REM Set the conda environment Python path
set CONDA_PYTHON=%USERPROFILE%\.conda\envs\cc\python.exe

REM Verify conda Python exists
if not exist "%CONDA_PYTHON%" (
    echo [ERROR] Conda environment Python not found at:
    echo         %CONDA_PYTHON%
    echo.
    echo Please ensure the 'cc' conda environment exists.
    echo Run these commands in a terminal:
    echo.
    echo     conda create -n cc python=3.10 -y
    echo     conda activate cc
    echo     pip install numpy scipy numba sounddevice wxPython nuitka
    echo.
    pause
    exit /b 1
)

echo [OK] Using Python: %CONDA_PYTHON%
echo.

REM Check if Nuitka is installed
"%CONDA_PYTHON%" -m nuitka --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Nuitka is not installed in the cc environment!
    echo Installing Nuitka...
    "%CONDA_PYTHON%" -m pip install nuitka
    if errorlevel 1 (
        echo [ERROR] Failed to install Nuitka
        pause
        exit /b 1
    )
)

echo [1/5] Nuitka version:
"%CONDA_PYTHON%" -m nuitka --version
echo.

REM Ensure C++ extension is compiled
echo [2/5] Checking C++ extension...
if not exist simplex5d.cp*.pyd (
    echo [WARNING] simplex5d extension not found, compiling...
    "%CONDA_PYTHON%" setup.py build_ext --inplace
    if errorlevel 1 (
        echo [ERROR] Failed to compile C++ extension
        pause
        exit /b 1
    )
)
for %%F in (simplex5d.cp*.pyd) do set PYD_FILE=%%F
echo [OK] C++ extension found: %PYD_FILE%
echo.

REM Check for required files
echo [3/5] Checking required files...
if not exist main.py (
    echo [ERROR] main.py not found!
    pause
    exit /b 1
)
if not exist SoundGenerator.py (
    echo [ERROR] SoundGenerator.py not found!
    pause
    exit /b 1
)
if not exist SoundManager.py (
    echo [ERROR] SoundManager.py not found!
    pause
    exit /b 1
)
if not exist frequencies.py (
    echo [ERROR] frequencies.py not found!
    pause
    exit /b 1
)
if not exist guide.html (
    echo [WARNING] guide.html not found - user guide will not be available
)
echo [OK] All required files present
echo.

REM Clean previous builds
echo [4/5] Cleaning previous builds...
if exist CrystalCare.exe del CrystalCare.exe
if exist CrystalCare.dist rmdir /s /q CrystalCare.dist
if exist CrystalCare.build rmdir /s /q CrystalCare.build
if exist CrystalCare.onefile-build rmdir /s /q CrystalCare.onefile-build
if exist main.dist rmdir /s /q main.dist
if exist main.build rmdir /s /q main.build
if exist main.onefile-build rmdir /s /q main.onefile-build
echo [OK] Cleaned
echo.

REM Start compilation
echo [5/5] Starting Nuitka compilation...
echo This may take 10-20 minutes depending on your system...
echo.

"%CONDA_PYTHON%" -m nuitka ^
  --standalone ^
  --onefile ^
  --enable-plugin=numpy ^
  --enable-plugin=anti-bloat ^
  --lto=yes ^
  --include-module=simplex5d ^
  --include-module=frequencies ^
  --include-module=SoundGenerator ^
  --include-module=SoundManager ^
  --include-package=numba ^
  --include-package=llvmlite ^
  --include-package=scipy ^
  --include-package=scipy.signal ^
  --include-package=scipy.ndimage ^
  --include-package=scipy.integrate ^
  --include-package=scipy.io ^
  --include-package=sounddevice ^
  --include-package=wx ^
  --include-data-files=%PYD_FILE%=%PYD_FILE% ^
  --windows-disable-console ^
  --assume-yes-for-downloads ^
  --show-progress ^
  --output-filename=CrystalCare.exe ^
  --company-name="CrystalCare" ^
  --product-name="CrystalCare" ^
  --product-version="1.0.0" ^
  --file-description="Sacred Frequency Generator" ^
  main.py

if errorlevel 1 (
    echo.
    echo ============================================================
    echo [ERROR] Compilation failed!
    echo ============================================================
    echo.
    echo Common fixes:
    echo 1. Ensure all dependencies are installed in cc environment
    echo 2. Run: "%CONDA_PYTHON%" -m pip install nuitka numba scipy sounddevice wxPython
    echo 3. Check that simplex5d.pyd was compiled correctly
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [SUCCESS] Compilation completed!
echo ============================================================
echo.
echo Executable: CrystalCare.exe
echo.
echo Size:
for %%A in (CrystalCare.exe) do echo   %%~zA bytes (%%~nxA)
echo.
echo ============================================================
echo Distribution Checklist:
echo ============================================================
echo.
echo The executable is self-contained. To distribute:
echo.
echo 1. Test CrystalCare.exe on this machine
echo 2. Test on a clean Windows machine (no Python installed)
echo 3. Verify all features work:
echo    - Play sound (all 7 frequency modes)
echo    - Save to WAV
echo    - Batch save
echo    - Open Guide (guide.html is embedded)
echo 4. For sessions over 60 seconds, verify sacred layers activate
echo.
pause
