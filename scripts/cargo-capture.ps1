# Run one cargo invocation with the pinned SDK environment and capture both streams.
# Why Start-Process: PowerShell surfaces cargo's stderr as NativeCommandError and aborts
# the pipeline; redirecting to files is the only stable way to capture it.
# NOTE: keep this file ASCII-only - Windows PowerShell 5.1 reads .ps1 as ANSI without a BOM.
param(
  [string]$Out = "",
  [string]$CargoArgs = "check -p hifishifter-plugin"
)
# Repo root = the parent of scripts/. Defaults and the working directory must
# stay relative to this file, not to whoever wrote the script last.
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ($Out -eq "") { $Out = Join-Path $RepoRoot ".tmp\cargo.txt" }
. "$PSScriptRoot\env\msvc-env.ps1"
. "$PSScriptRoot\env\sdk-env.ps1"
Set-SdkEnvironment
$exe = (Get-Command cargo).Source
$stdout = "$Out.stdout"
$stderr = "$Out.stderr"
$argv = $CargoArgs -split "\s+" | Where-Object { $_ -ne "" }
$process = Start-Process -FilePath $exe -ArgumentList $argv `
  -WorkingDirectory (Join-Path $RepoRoot "backend") `
  -NoNewWindow -Wait -PassThru `
  -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Get-Content $stdout, $stderr -ErrorAction SilentlyContinue | Set-Content -Encoding utf8 $Out
Write-Output "exit=$($process.ExitCode)"
