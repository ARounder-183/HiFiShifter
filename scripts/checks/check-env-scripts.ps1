# 点源的 env 脚本只允许改 $env:* —— 不允许在调用者作用域里留下普通变量。
#
# 【为什么需要这条门禁】真实事故：scripts/env/msvc-env.ps1 里有一个
# `foreach ($root in Get-VsInstallRoots)`，而它是**点源**的 —— 于是调用者（构建脚本）
# 的 `$root`（仓库根）被覆盖成 Visual Studio 安装目录，后续每个 `Join-Path $root ...`
# 都拼到 `C:\Program Files\Microsoft Visual Studio\...` 下面。症状是构建在
# "找不到 models.json"处失败，而报错里只有那个荒谬的路径 —— 没有任何线索指向"变量被
# 前一份被点源的脚本改写"。
#
# 【判据】在子进程里给一组常见名字放哨兵值，点源 env 脚本，再读回来。任何一个变了
# 就说明该脚本又把临时变量写进了调用者作用域。
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$probe = Join-Path ([IO.Path]::GetTempPath()) ('hfs-env-scope-' + [Guid]::NewGuid().ToString('N') + '.ps1')
$names = @('root', 'lines', 'dump', 'vcvars', 'dist', 'profile', 'stage', 'candidate', 'found')
# 哨兵与校验循环都必须从 $names 生成：此前两处各硬编码一份，$names 本身从未被读
# —— 往 $names 里加新哨兵名不会生效，门禁会悄悄少测一个变量。
# （@() 里各语句换行分隔：管道输出的元素会被摊平进数组，逗号分隔则会把嵌套
# 数组留成单个元素 —— WriteAllLines 会把它拼成一行。）
$body = @(
    $names | ForEach-Object { ('$' + $_ + ' = "SENTINEL-' + $_ + '"') }
    # 清掉"已导入 MSVC"的证据，让导入器的真实分支跑一遍（否则它早退，测不到泄漏）。
    '$env:INCLUDE = ""'
    '$env:LIB = ""'
    '$env:LIBPATH = ""'
    '$env:VSCMD_VER = ""'
    ('$ErrorActionPreference = "Stop"')
    ('. "' + (Join-Path $root 'scripts\env\msvc-env.ps1') + '"')
    ('. "' + (Join-Path $root 'scripts\env\sdk-env.ps1') + '"')
    'Set-SdkEnvironment'
    ('foreach ($name in ' + (($names | ForEach-Object { '"' + $_ + '"' }) -join ',') + ') {')
    '  $value = Get-Variable -Name $name -ValueOnly -ErrorAction SilentlyContinue'
    '  Write-Output ($name + "=" + $value)'
    '}'
)
[IO.File]::WriteAllLines($probe, $body, [Text.Encoding]::UTF8)
try {
    $pwsh = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
    if (!$pwsh) { $pwsh = (Get-Process -Id $PID).Path }
    $output = & $pwsh -NoProfile -ExecutionPolicy Bypass -File $probe 2>&1
    if ($LASTEXITCODE) { throw "探针进程失败（退出码 $LASTEXITCODE）：$($output -join ' ')" }
}
finally {
    Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
}
$leaked = @()
foreach ($line in $output) {
    if ($line -notmatch '^([a-z]+)=(.*)$') { continue }
    $name = $Matches[1]
    $value = $Matches[2]
    if ($value -ne "SENTINEL-$name") { $leaked += "$name=$value" }
}
if ($leaked.Count -gt 0) {
    throw ("这些变量被点源的 env 脚本改写了（它们必须留成函数的局部变量）：" + ($leaked -join ', '))
}
Write-Host 'env scope gate OK: dot-sourced scripts only touch env vars'
