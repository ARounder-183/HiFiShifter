# Rust 门禁的唯一入口：fmt / clippy / test，以及唯一的跳过清单。
#
# 【为什么要有这个脚本】同一组 cargo 命令此前在三条工作流里各写一遍：build.yml、
# vst3.yml、vst3-unix.yml。它们的参数（--locked / --offline / --jobs / --skip /
# --test-threads）互不相同，也没有任何东西保证它们等价 —— 于是一条用例可以在一条
# 流水线上绿、在另一条上红，而"哪条才是判据"没人说得清。
# 现在工作流只调用本脚本：**判据只有一处**，-PrintCommand 打印出它解析到的命令行。
#
#   .\scripts\checks\run-rust-gates.ps1                     # pr：fmt + clippy + test
#   .\scripts\checks\run-rust-gates.ps1 -Only fmt           # 只跑格式
#   .\scripts\checks\run-rust-gates.ps1 -Profile full        # 追加 no-default-features / 并发测试
#   .\scripts\checks\run-rust-gates.ps1 -PrintCommand        # 只打印命令行（本地与 CI 对照）
[CmdletBinding()]
param(
    [ValidateSet('pr', 'full')][string]$Profile = 'pr',
    [ValidateSet('all', 'fmt', 'clippy', 'test')][string]$Only = 'all',
    # 空 = 整个 workspace；指定包名时只跑那一个 crate（离线校验用）。
    [string]$Package = '',
    [ValidateSet('serial', 'parallel')][string]$TestMode = 'serial',
    # 显式覆盖平台（默认按运行器判定）：跳过清单里有 [unix] 专用条目。
    [ValidateSet('', 'windows', 'unix')][string]$Platform = '',
    [switch]$PrintCommand
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $root

if (!$Platform) {
    if ($env:RUNNER_OS -eq 'Windows') { $Platform = 'windows' }
    elseif ($env:RUNNER_OS) { $Platform = 'unix' }
    elseif ([IO.Path]::DirectorySeparatorChar -eq '\') { $Platform = 'windows' }
    else { $Platform = 'unix' }
}

# ── 跳过清单 ───────────────────────────────────────────────────────────────
# 唯一来源是 scripts/checks/test-skips.txt：工作流里不再出现任何 --skip。
function Get-Skips {
    $file = Join-Path $PSScriptRoot 'test-skips.txt'
    if (!(Test-Path $file)) { throw "缺少跳过清单：$file" }
    $names = New-Object System.Collections.Generic.List[string]
    foreach ($line in Get-Content -LiteralPath $file -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (!$trimmed) { continue }
        if ($trimmed.StartsWith('#')) { continue }
        $entryPlatform = ''
        if ($trimmed.StartsWith('[unix]')) { $entryPlatform = 'unix'; $trimmed = $trimmed.Substring(6).Trim() }
        elseif ($trimmed.StartsWith('[windows]')) { $entryPlatform = 'windows'; $trimmed = $trimmed.Substring(9).Trim() }
        if ($entryPlatform -and $entryPlatform -ne $Platform) { continue }
        # 条目形如 "<测试名> — <跟踪位置> — <原因>"：只取第一段。
        $name = ($trimmed -split ' — ')[0].Trim()
        if ($name) { $names.Add($name) }
    }
    return $names
}

$manifest = 'backend/Cargo.toml'
$skipArgs = @()
foreach ($name in Get-Skips) { $skipArgs += @('--skip', $name) }

function Get-CargoCommand([string]$kind) {
    # 【为什么逐条 Write-Output 而不是 return @(...)】PowerShell 的 return 会把数组
    # 原样塞进输出流的一个对象里，调用方 foreach 只会拿到"一个数组"（打印出来是空格
    # 拼接的一行）。逐条写出去，管道才会自然展开成多条命令。
    $msvc = Join-Path $root 'scripts\env\msvc-env.ps1'
    $prefix = ''
    if (Test-Path $msvc) { $prefix = ". $msvc;" }
    switch ($kind) {
        'fmt' {
            # hfs-gui（VST3 的独立渲染进程）在 backend workspace 之外，--all 覆盖不到它；
            # 它曾因此积累 rustfmt 漂移，而"能编译"看不出这种退化。
            Write-Output "$prefix cargo fmt --manifest-path $manifest --all --check"
            Write-Output "$prefix cargo fmt --manifest-path backend/hfs-gui/Cargo.toml --all --check"
        }
        'clippy' {
            $target = if ($Package) { "-p $Package" } else { '--workspace' }
            Write-Output "$prefix cargo clippy --locked --manifest-path $manifest $target --all-targets -- -D warnings"
            if ($Profile -eq 'full' -and !$Package) {
                # 纯 Rust 形态（无 onnx / vslib）从未被编译过：未来做轻量形态或重排 cfg
                # 门控时，破坏只会先在用户处爆发。分钟级成本，单独钉住。
                Write-Output "$prefix cargo clippy --locked --manifest-path $manifest -p hifishifter-kernel --no-default-features --all-targets -- -D warnings"
            }
        }
        'test' {
            $target = if ($Package) { "-p $Package" } else { '--workspace' }
            $threads = if ($TestMode -eq 'parallel') { '4' } else { '1' }
            $skips = if ($skipArgs.Count -gt 0) { ' ' + ($skipArgs -join ' ') } else { '' }
            Write-Output "$prefix cargo test --locked --manifest-path $manifest $target -- --test-threads=$threads$skips"
        }
    }
}

$kinds = switch ($Only) {
    'all' { @('fmt', 'clippy', 'test') }
    default { @($Only) }
}

if ($PrintCommand) {
    foreach ($kind in $kinds) {
        # 【为什么要 @(...)】PowerShell 会把只有一条命令的数组"降级"成字符串，
        # 直接 foreach 只会打印一行、而 [0] 会取到第一个字符 —— 两者都不会报错，
        # 只会让自证输出看起来"少了几条门禁"。
        foreach ($command in @(Get-CargoCommand $kind)) { Write-Output $command }
    }
    if ($Profile -eq 'full' -and ($Only -eq 'all' -or $Only -eq 'test') -and !$Package) {
        # full 档多一条并发复跑：--test-threads=1 会掩盖竞态，而插件最容易出事的地方
        # （重算与自动应用的版本竞态、快照发布与读者回收）全部是并发敏感的。
        Write-Output (@(Get-CargoCommand 'test')[0] -replace '--test-threads=1', '--test-threads=4')
    }
    exit 0
}

foreach ($kind in $kinds) {
    foreach ($command in @(Get-CargoCommand $kind)) {
        Write-Host "== $kind" -ForegroundColor Cyan
        Write-Host "   $command" -ForegroundColor DarkGray
        Invoke-Expression $command
        if ($LASTEXITCODE) { throw "cargo $kind failed (exit $LASTEXITCODE)" }
    }
}

if ($Profile -eq 'full' -and ($Only -eq 'all' -or $Only -eq 'test') -and !$Package) {
    Write-Host '== test (concurrent)' -ForegroundColor Cyan
    $concurrent = @(Get-CargoCommand 'test')[0] -replace '--test-threads=1', '--test-threads=4'
    Write-Host "   $concurrent" -ForegroundColor DarkGray
    Invoke-Expression $concurrent
    if ($LASTEXITCODE) { throw "concurrent cargo test failed (exit $LASTEXITCODE)" }
}

Write-Host 'All Rust gates passed.' -ForegroundColor Green
