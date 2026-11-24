param([ValidateSet('edgevision','zacatecas_videos')][string]$Source = 'edgevision')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$lock = Get-Content -LiteralPath (Join-Path $projectRoot 'data/sources.lock.json') -Raw | ConvertFrom-Json
$item = $lock.sources | Where-Object id -eq $Source
$folder = Join-Path $projectRoot 'data/sources'
[System.IO.Directory]::CreateDirectory($folder) | Out-Null
$archive = Join-Path $folder "$Source-v$($item.version).zip"
$partial = "$archive.part"
if (-not (Test-Path -LiteralPath $archive)) {
  # Resume is allowed only when the server supports it. curl refuses invalid ranges.
  & curl.exe --location --fail --retry 2 --connect-timeout 30 --max-time 1800 --continue-at - --user-agent 'Mozilla/5.0' $item.download --output $partial
  if ($LASTEXITCODE -ne 0) { throw "Download incomplete. Partial retained at $partial; rerun to resume." }
  # Validate the ZIP directory before marking a download complete.
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $zip = [System.IO.Compression.ZipFile]::OpenRead($partial)
  $zip.Dispose()
  Move-Item -LiteralPath $partial -Destination $archive
}
$receipt = [ordered]@{
  source = $Source; version = $item.version; url = $item.download
  downloadedAtUtc = [DateTime]::UtcNow.ToString('o')
  bytes = (Get-Item -LiteralPath $archive).Length
  sha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
  note = 'Local integrity fingerprint, not compared against a publisher-supplied checksum.'
}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectRoot "reports/$Source-download.json") -Encoding utf8
Write-Output $archive
