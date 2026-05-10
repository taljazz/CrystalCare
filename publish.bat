@echo off
rem CrystalCare publish script — produces a single self-contained .exe at the
rem project root, ready for package.bat to wrap into CrystalCare.zip.
rem
rem Why output to the project root (not the default deep publish folder):
rem  * Default dotnet publish writes to src\CrystalCare\bin\Release\
rem    net8.0-windows\win-x64\publish\CrystalCare.exe — buried four levels deep.
rem  * package.bat expects CrystalCare.exe at the PROJECT ROOT to zip it up.
rem  * The legacy Nuitka build.bat used to put it at the root for that reason.
rem  * Outputting here ('-o "%~dp0"') keeps the workflow continuous: run
rem    publish.bat → CrystalCare.exe at root → run package.bat → CrystalCare.zip.
rem
rem Why each MSBuild flag matters:
rem  * PublishSingleFile=true               → bundles all DLLs into the EXE
rem  * IncludeNativeLibrariesForSelfExtract → bundles native libs (NAudio)
rem  * EnableCompressionInSingleFile        → ~30% smaller bundle
rem  * DebugType=embedded                   → folds .pdb into the EXE so no
rem                                            separate symbol files leak out
rem  * -o "%~dp0"                           → project-root output directory

cd /d "%~dp0"

echo ============================================================
echo CrystalCare - Publishing Single Executable
echo ============================================================
echo.

rem Remove any stale CrystalCare.exe at the project root before publishing.
rem Without this, a failed publish would leave the previous build in place
rem and the user might think the new run "didn't update."
if exist CrystalCare.exe (
    echo Removing previous CrystalCare.exe...
    del /f /q CrystalCare.exe
)

echo Running dotnet publish...
echo.

dotnet publish src\CrystalCare\CrystalCare.csproj ^
    -c Release ^
    -r win-x64 ^
    --self-contained true ^
    -p:PublishSingleFile=true ^
    -p:IncludeNativeLibrariesForSelfExtract=true ^
    -p:EnableCompressionInSingleFile=true ^
    -p:DebugType=embedded ^
    -o "%~dp0"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================
    echo [ERROR] Build failed.
    echo ============================================================
    pause
    exit /b 1
)

if not exist CrystalCare.exe (
    echo.
    echo ============================================================
    echo [ERROR] Publish completed but CrystalCare.exe was not produced
    echo         at the project root. Check the dotnet output above.
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [SUCCESS] Single-file build complete.
echo ============================================================
echo.
echo Output: %~dp0CrystalCare.exe
for %%A in (CrystalCare.exe) do echo Size:   %%~zA bytes
echo.
echo Next step: run package.bat to wrap into CrystalCare.zip
echo.
pause
