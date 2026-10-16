# Run one cargo invocation with the pinned SDK environment and capture both streams.
# Why Start-Process: PowerShell surfaces cargo's stderr as NativeCommandError and aborts
# the pipeline; redirecting to files is the only stable way to capture it.
# NOTE: keep this file ASCII-only - Windows PowerShell 5.1 reads .ps1 as ANSI without a BOM.
param(
  [string]$Out = "D:\Projects\HiFiShifter\.tmp\cargo.txt",
  [string]$CargoArgs = "check -p hifishifter-plugin"
)
. "$PSScriptRoot\env\msvc-env.ps1"
. "$PSScriptRoot\env\sdk-env.ps1"
Set-SdkEnvironment
$exe = (Get-Command cargo).Source
$stdout = "$Out.stdout"
$stderr = "$Out.stderr"
$argv = $CargoArgs -split "\s+" | Where-Object { $_ -ne "" }
$process = Start-Process -FilePath $exe -ArgumentList $argv `
  -WorkingDirectory "D:\Projects\HiFiShifter\backend" `
  -NoNewWindow -Wait -PassThru `
  -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Get-Content $stdout, $stderr -ErrorAction SilentlyContinue | Set-Content -Encoding utf8 $Out
Write-Output "exit=$($process.ExitCode)"
