param([string]$Repo = ".")

$Canonical = $env:UV_AUTO_PYTHON
if (-not $Canonical -or -not (Test-Path $Canonical)) {
  $Canonical = "D:\tanul\it\uv\uv-auto-python.ps1"
}
$local = Join-Path $PSScriptRoot "uv-auto-python.local.ps1"

if (-not (Test-Path $Canonical)) {
  if (Test-Path $local) { $Canonical = $local }
}

if (-not (Test-Path $Canonical)) {
  Write-Error "uv-auto-python.ps1 not found. Checked:`n  $env:UV_AUTO_PYTHON`n  D:\tanul\it\uv\uv-auto-python.ps1`n  $local"
  exit 1
}

& $Canonical -Repo $Repo
