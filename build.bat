@echo off
chcp 65001 >nul
cd /d "%~dp0"
where pwsh >nul 2>&1 && (set "HFS_PS=pwsh") || (set "HFS_PS=powershell")
%HFS_PS% -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build\build.ps1" %*
if errorlevel 1 (
    echo.
    echo 构建失败。
    pause
    exit /b 1
)
echo.
echo 完成。产物在 dist 目录下。
pause
