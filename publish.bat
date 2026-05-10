@echo off
rem CrystalCare publish script — produces a single self-contained .exe at the
rem project root, ready for package.bat to wrap into CrystalCare.zip.
rem
rem Why publish to a subfolder then COPY to root (instead of -o "%~dp0"):
rem  * Publishing directly into the project root breaks single-file bundling
rem    in this SDK. The published EXE silently fails to load managed code —
rem    the process starts, no window appears, no exception is thrown, and the
rem    static class constructor never fires. Diagnosed by trace-logging into
rem    %TEMP% from App's static ctor: log was empty when -o pointed at root,
rem    populated when -o pointed at a subfolder. Likely the bundler gets
rem    confused by other files in the root (the .sln, source folders, etc).
rem  * Publishing to "publish-build/" subfolder produces a working EXE.
rem    We then COPY just CrystalCare.exe to the project root so package.bat
rem    can find it there (legacy Nuitka build.bat behavior).
rem
rem Why each MSBuild flag matters:
rem  * PublishSingleFile=true               → bundles all DLLs into the EXE
rem  * IncludeNativeLibrariesForSelfExtract → bundles native libs (NAudio
rem      Wasapi etc). Cannot use IncludeAllContentForSelfExtract — that
rem      ALSO breaks bundling for this WPF + WinForms app on this SDK.
rem  * EnableCompressionInSingleFile        → ~30%% smaller bundle
rem  * DebugType=embedded                   → folds .pdb into the EXE so no
rem                                            separate symbol files leak out

cd /d "%~dp0"

echo ============================================================
echo CrystalCare - Publishing Single Executable
echo ============================================================
echo.

rem Remove any stale CrystalCare.exe at the project root before publishing.
if exist CrystalCare.exe (
    echo Removing previous CrystalCare.exe...
    del /f /q CrystalCare.exe
)

rem Clean the intermediate publish folder so each run is reproducible.
if exist publish-build (
    echo Cleaning previous publish-build folder...
    rmdir /s /q publish-build
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
    -o "%~dp0publish-build"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================
    echo [ERROR] Build failed.
    echo ============================================================
    pause
    exit /b 1
)

if not exist "publish-build\CrystalCare.exe" (
    echo.
    echo ============================================================
    echo [ERROR] Publish completed but CrystalCare.exe was not produced
    echo         in publish-build\. Check the dotnet output above.
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo Copying CrystalCare.exe to project root...
copy /y "publish-build\CrystalCare.exe" "CrystalCare.exe" >nul

if not exist CrystalCare.exe (
    echo [ERROR] Failed to copy CrystalCare.exe to project root.
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
