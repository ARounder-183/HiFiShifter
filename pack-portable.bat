@echo off
chcp 65001 >nul 2>&1
title HiFiShifter 便携版打包
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\pack-portable.ps1" %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo 打包失败，按任意键退出...
    pause >nul
) else (
    echo.
    echo 按任意键退出...
    pause >nul
)
