# 【已搬迁】插件 bundle 构建已提升为产品脚本 `tools/build-plugin-bundle.ps1`。
#
# 保留本文件是为了让 `probe/ara/` 下的历史运行记录（EMBEDDED-EDITOR-RUN.md 等）
# 里写的命令仍然可用；它不含任何逻辑，只做转发。新代码请直接调用 tools/ 下的那份 ——
# 本目录按 `probe/ara/README.md` 的说明是可整体删除的一次性探针产物。
[CmdletBinding()]
param(
    [switch]$SkipFrontend,
    [switch]$Release,
    [ValidatePattern('^[a-z0-9][a-z0-9-]{0,63}$')][string]$BundleDirectory = 'embedded-vst3'
)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '..\..\tools\build-plugin-bundle.ps1') `
    -SkipFrontend:$SkipFrontend -Release:$Release -BundleDirectory $BundleDirectory
if ($LASTEXITCODE -ne 0) { throw "tools/build-plugin-bundle.ps1 failed with exit code $LASTEXITCODE" }
