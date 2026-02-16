@echo off
REM CrystalCare Package Script
REM Zips CrystalCare.exe and guide.html into CrystalCare.zip

echo ============================================================
echo CrystalCare - Packaging Script
echo ============================================================
echo.

cd /d "%~dp0"

REM Check that required files exist
echo Checking required files...

if not exist CrystalCare.exe (
    echo [ERROR] CrystalCare.exe not found!
    echo.
    echo Run build.bat first to compile the executable.
    pause
    exit /b 1
)

if not exist guide.html (
    echo [ERROR] guide.html not found!
    echo.
    echo The user guide is required for distribution.
    pause
    exit /b 1
)

echo [OK] CrystalCare.exe found
echo [OK] guide.html found
echo.

REM Delete existing zip if present
if exist CrystalCare.zip (
    echo Removing existing CrystalCare.zip...
    del CrystalCare.zip
    if exist CrystalCare.zip (
        echo [ERROR] Failed to delete existing CrystalCare.zip
        echo.
        echo The file may be open in another program.
        pause
        exit /b 1
    )
    echo [OK] Removed old CrystalCare.zip
    echo.
)

REM Create staging directory for flat zip structure
echo Creating CrystalCare.zip...
if exist _package rmdir /s /q _package
mkdir _package
copy CrystalCare.exe _package\ >nul
copy guide.html _package\ >nul

REM Create zip using PowerShell Compress-Archive
powershell -NoProfile -Command "Compress-Archive -Path '_package\*' -DestinationPath 'CrystalCare.zip' -Force"
set ZIP_ERROR=%errorlevel%

REM Clean up staging directory
rmdir /s /q _package

if %ZIP_ERROR% neq 0 (
    echo.
    echo [ERROR] Failed to create CrystalCare.zip
    echo.
    echo Ensure PowerShell is available on this system.
    pause
    exit /b 1
)

if not exist CrystalCare.zip (
    echo.
    echo [ERROR] CrystalCare.zip was not created
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [SUCCESS] CrystalCare.zip created!
echo ============================================================
echo.
echo Contents:
echo   - CrystalCare.exe
echo   - guide.html
echo.
echo Size:
for %%A in (CrystalCare.zip) do echo   %%~zA bytes
echo.
pause
