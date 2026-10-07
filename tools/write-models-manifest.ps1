# 生成模型清单 `models.json`：版本号 + 每个文件的字节数与 sha256。
#
# 【为什么版本号由内容派生】模型换一版就必须重建共享库，否则新代码会读到旧权重 ——
# 那种错误不报错，只让推理结果悄悄变差。让版本号等于内容哈希，"模型变了但版本号忘了
# 改"这件事就不可能发生，也不需要人去维护一个版本字符串。
#
# 【谁读它】独立 App 与 ARA 插件在启动时都会读自己 bundle 里的这份清单：
# - 判断共享模型库里的那一份是否可用（版本一致 + 文件齐全）；
# - 决定是否从 bundle 建立共享库（见 backend/hifishifter-kernel/src/model_store.rs）。
#
# 用法：
#   pwsh -File tools/write-models-manifest.ps1            # 生成
#   pwsh -File tools/write-models-manifest.ps1 -Check     # 只校验是否最新（CI 用）
[CmdletBinding()]
param([switch]$Check)

$ErrorActionPreference = 'Stop'
$manifestRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$manifestModels = Join-Path $manifestRoot 'backend\src-tauri\resources\models'
$manifestPath = Join-Path $manifestModels 'models.json'

# 只登记真正参与推理的文件。`coreml_pads_patch.txt` 是 macOS 构建期的补丁输入、
# `config.json` / `config.yaml` 是模型自带的配置 —— 它们一起改才叫"换模型"，
# 因此一并纳入版本计算，但目录里的临时产物不纳入。
$manifestExcluded = @('models.json', 'coreml_pads_patch.txt')

$manifestFiles = @()
foreach ($manifestFile in Get-ChildItem -LiteralPath $manifestModels -Recurse -File |
    Where-Object { $_.Name -notin $manifestExcluded } | Sort-Object FullName) {
    $manifestRelative = $manifestFile.FullName.Substring($manifestModels.Length + 1).Replace('\', '/')
    $manifestFiles += [ordered]@{
        path   = $manifestRelative
        bytes  = $manifestFile.Length
        sha256 = (Get-FileHash -LiteralPath $manifestFile.FullName -Algorithm SHA256).Hash
    }
}
if (!$manifestFiles.Count) { throw "No model files found under $manifestModels" }

# 版本 = 所有 (路径, sha256) 汇总后的哈希前 12 位。路径参与计算，因此"改名"也算变更。
$manifestLines = @($manifestFiles | ForEach-Object { "$($_.path) $($_.sha256)" })
$manifestHasher = [Security.Cryptography.SHA256]::Create()
try {
    $manifestVersion = ([BitConverter]::ToString($manifestHasher.ComputeHash(
        [Text.Encoding]::UTF8.GetBytes(($manifestLines -join "`n"))))).Replace('-', '').Substring(0, 12).ToLowerInvariant()
} finally { $manifestHasher.Dispose() }

$manifestJson = [ordered]@{ version = $manifestVersion; files = $manifestFiles } | ConvertTo-Json -Depth 5
# 与仓库其它生成物一致：UTF-8 无 BOM、LF 换行、结尾一个换行。
$manifestText = ($manifestJson -replace "`r`n", "`n") + "`n"

if ($Check) {
    if (!(Test-Path -LiteralPath $manifestPath)) { throw "Missing $manifestPath; run tools/write-models-manifest.ps1" }
    $manifestCurrent = (Get-Content -LiteralPath $manifestPath -Raw) -replace "`r`n", "`n"
    if ($manifestCurrent -ne $manifestText) {
        throw "models.json is stale; run tools/write-models-manifest.ps1"
    }
    Write-Output "models.json is up to date (version $manifestVersion)"
    return
}

[IO.File]::WriteAllText($manifestPath, $manifestText, [Text.UTF8Encoding]::new($false))
Write-Output "wrote $manifestPath (version $manifestVersion, $($manifestFiles.Count) files)"
