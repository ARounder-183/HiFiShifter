# Validate UI resources independently of the native module load probe.
[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$FrontendDirectory)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($FrontendDirectory).TrimEnd('\')
$prefix = $root + '\'
foreach ($entry in 'plugin.html', 'detached.html') {
    if (!(Test-Path -LiteralPath (Join-Path $root $entry) -PathType Leaf)) {
        throw "Plugin frontend entry missing: $entry ($root)"
    }
}
$assets = @(Get-ChildItem -LiteralPath (Join-Path $root 'assets') -File -ErrorAction Stop)
if (!$assets.Count) { throw "Plugin frontend assets are empty: $root" }
$checked = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
function Assert-LocalResource([string]$reference, [string]$fromDirectory) {
    if ($reference -match '^(?:[a-z][a-z0-9+.-]*:|//|#)') { return }
    $reference = ($reference -split '[?#]', 2)[0]
    if (!$reference) { return }
    $base = if ($reference.StartsWith('/')) { $root } else { $fromDirectory }
    $path = [IO.Path]::GetFullPath((Join-Path $base $reference.TrimStart('/')))
    if (!$path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Frontend reference escapes its resource root: $reference"
    }
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Plugin frontend resource missing: $reference (from $fromDirectory)"
    }
    [void]$checked.Add($path)
}
foreach ($html in (Get-ChildItem -LiteralPath $root -Filter '*.html' -File)) {
    $source = Get-Content -LiteralPath $html.FullName -Raw
    foreach ($match in [regex]::Matches($source, '(?:src|href)\s*=\s*["'']([^"'']+)["'']')) {
        Assert-LocalResource $match.Groups[1].Value $root
    }
}
foreach ($js in ($assets | Where-Object Extension -eq '.js')) {
    $source = Get-Content -LiteralPath $js.FullName -Raw
    # Bundled imports and Vite's generated dependency table use literal paths.
    foreach ($match in [regex]::Matches($source, '["'']((?:\./|\.\./|assets/)[^"''\s]+\.(?:js|css|wasm|png|svg|ico|woff2?))["'']')) {
        $reference = $match.Groups[1].Value
        $base = if ($reference.StartsWith('assets/')) { $root } else { $js.DirectoryName }
        Assert-LocalResource $reference $base
    }
}
Write-Output "Frontend resources verified: $($checked.Count) references ($root)"
