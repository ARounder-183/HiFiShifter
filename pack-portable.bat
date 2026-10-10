@echo off
chcp 65001 >nul 2>&1
title HiFiShifter App / VST3 打包
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\pack-portable.ps1" %*
set "HFS_EXITCODE=%ERRORLEVEL%"
echo.
if not "%HFS_EXITCODE%"=="0" (
    echo 打包失败（退出码 %HFS_EXITCODE%）。请查看上面的错误信息后重试。
) else (
    echo 打包完成，产物在 dist\ 目录下。
)
echo.
echo 按任意键退出...
pause >nul
exit /b %HFS_EXITCODE%
