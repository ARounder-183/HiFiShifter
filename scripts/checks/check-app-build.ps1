# A release profile alone does not enable Tauri's embedded frontend. Probe the
# executable itself so a localhost-dependent App can never pass packaging.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ExecutablePath
)

$ErrorActionPreference = 'Stop'
$executable = (Resolve-Path -LiteralPath $ExecutablePath).Path
$probeRoot = Join-Path ([IO.Path]::GetTempPath()) ('hfs-app-build-check-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path (Join-Path $probeRoot 'Local'), (Join-Path $probeRoot 'Roaming') -Force | Out-Null
$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = $executable
$start.Arguments = '--print-build-info'
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
$start.EnvironmentVariables['LOCALAPPDATA'] = Join-Path $probeRoot 'Local'
$start.EnvironmentVariables['APPDATA'] = Join-Path $probeRoot 'Roaming'
$probe = [Diagnostics.Process]::new()
$probe.StartInfo = $start
try {
    if (!$probe.Start()) { throw 'Could not start the App build probe' }
    if (!$probe.WaitForExit(15000)) {
        $probe.Kill()
        throw 'App did not answer --print-build-info; it must not launch a GUI during this check'
    }
    $output = $probe.StandardOutput.ReadToEnd()
    $detail = $probe.StandardError.ReadToEnd()
    if ($probe.ExitCode -ne 0) { throw "App build probe failed ($($probe.ExitCode)): $detail" }
    $info = $output | ConvertFrom-Json
    if ($info.product -ne 'HiFiShifter' -or $info.ui_source -ne 'embedded' -or $info.custom_protocol -ne $true -or $info.debug_assertions -ne $false) {
        throw "App is not a production build with an embedded frontend: $output. Use cargo tauri build or cargo build --release --features custom-protocol."
    }
    Write-Output "App build verified: embedded frontend, custom protocol enabled ($($info.version))"
}
finally {
    $probe.Dispose()
}
