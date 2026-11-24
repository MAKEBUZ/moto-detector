param([int]$Connections = 6)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$url = 'https://data.mendeley.com/public-api/zip/j82bnw7gsr/download/1'
$total = 1490218052 # verified Content-Range for version 1, 2026-11-23
$folder = Join-Path $root 'data/sources/edgevision-parts'
[System.IO.Directory]::CreateDirectory($folder) | Out-Null
$partial = Join-Path $root 'data/sources/edgevision-v1.zip.part'
$archive = Join-Path $root 'data/sources/edgevision-v1.zip'
if (Test-Path -LiteralPath $archive) { throw 'Completed archive already exists; use its receipt.' }
if ($Connections -lt 1 -or $Connections -gt 8) { throw 'Use between one and eight connections.' }
$prefixLength = if (Test-Path -LiteralPath $partial) { (Get-Item -LiteralPath $partial).Length } else { 0 }
if ($prefixLength -gt $total) { throw 'Existing partial is larger than expected.' }
$chunk = [long](4MB)
$ranges = @()
for ($i = 0; $prefixLength + $i * $chunk -lt $total; $i++) {
  $start = $prefixLength + $i * $chunk
  if ($start -ge $total) { break }
  $end = [Math]::Min($total - 1, $start + $chunk - 1)
  $ranges += [pscustomobject]@{ Start=$start; End=$end; Path=(Join-Path $folder "$start-$end.bin") }
}
$results = @($ranges | ForEach-Object -Parallel {
  $r = $_
  $expected = $r.End - $r.Start + 1
  if ((Test-Path -LiteralPath $r.Path) -and (Get-Item -LiteralPath $r.Path).Length -eq $expected) { return $r.Path }
  & curl.exe --silent --show-error --location --fail --retry 2 --connect-timeout 30 --max-time 90 --range "$($r.Start)-$($r.End)" $using:url --output $r.Path
  if ($LASTEXITCODE -ne 0) { throw "Download failed for range $($r.Start)-$($r.End)" }
  if ((Get-Item -LiteralPath $r.Path).Length -ne $expected) { throw 'Unexpected range length; refusing assembly.' }
  $r.Path
} -ThrottleLimit $Connections)
if ($results.Count -ne $ranges.Count) { throw 'Incomplete ranges; rerun to retry missing parts.' }
$combined = "$archive.assembling"
$out = [System.IO.File]::Create($combined)
try {
  $files = @()
  if ($prefixLength -gt 0) {
    if ((Get-Item -LiteralPath $partial).Length -ne $prefixLength) { throw 'Partial changed while downloading.' }
    $files += $partial
  }
  $files += @($ranges | ForEach-Object Path)
  foreach ($file in $files) {
    $inputStream = [System.IO.File]::OpenRead($file)
    try { $inputStream.CopyTo($out) } finally { $inputStream.Dispose() }
  }
} finally { $out.Dispose() }
if ((Get-Item -LiteralPath $combined).Length -ne $total) { throw 'Unexpected assembled length.' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($combined)
$zip.Dispose()
Move-Item -LiteralPath $combined -Destination $archive
& (Join-Path $PSScriptRoot 'download-public-source.ps1') -Source edgevision
