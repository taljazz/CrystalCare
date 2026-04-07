@echo off
echo Publishing CrystalCare as single executable...
dotnet publish src\CrystalCare\CrystalCare.csproj -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true -p:EnableCompressionInSingleFile=true
if %ERRORLEVEL% NEQ 0 (
    echo Build failed.
    pause
    exit /b 1
)
echo.
echo Build complete: src\CrystalCare\bin\Release\net8.0-windows\win-x64\publish\CrystalCare.exe
pause
