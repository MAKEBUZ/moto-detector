param([ValidateSet('motorcycle_helmet_od','mapua_helmet','ldrago_helmet','proper_helmet_usage','helmet_violation')][string]$Source = 'motorcycle_helmet_od', [switch]$MetadataOnly)
$ErrorActionPreference = 'Stop'
$root=Split-Path $PSScriptRoot -Parent
$key=$env:ROBOFLOW_API_KEY
if (-not $key -and (Test-Path -LiteralPath (Join-Path $root '.env'))) {
  foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $root '.env'))) {
    if ($line -match '^\s*ROBOFLOW_API_KEY\s*=\s*(.+?)\s*$') { $key=$Matches[1].Trim().Trim('"',"'") }
  }
}
if (-not $key) { throw 'Configure ROBOFLOW_API_KEY in the local .env file; never print the key.' }
$lock=Get-Content -LiteralPath (Join-Path $root 'data/sources.lock.json') -Raw | ConvertFrom-Json
$sourceInfo=$lock.sources | Where-Object id -eq $Source
$client=[System.Net.Http.HttpClient]::new()
$client.Timeout=[TimeSpan]::FromMinutes(30)
$encodedKey=[Uri]::EscapeDataString($key)
function Get-PrivateJson([string]$url) {
  $response=$client.GetAsync($url).GetAwaiter().GetResult()
  try {
    if (-not $response.IsSuccessStatusCode) { throw "Roboflow API request failed (HTTP $([int]$response.StatusCode)); URL and credentials suppressed." }
    return ($response.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json)
  } finally { $response.Dispose() }
}
try {
  $base="https://api.roboflow.com/$($sourceInfo.workspace)/$($sourceInfo.project)"
  $info=Get-PrivateJson "$($base)?api_key=$encodedKey"
  $version=@($info.versions | Where-Object { ([string]$_.id).Split('/')[-1] -eq [string]$sourceInfo.version })
  if ($version.Count -ne 1) { throw "Pinned version $($sourceInfo.version) not available for $Source; do not select latest automatically." }
  $receipt=[ordered]@{ source=$Source; version=$sourceInfo.version; checkedAtUtc=[DateTime]::UtcNow.ToString('o'); page="https://universe.roboflow.com/$($sourceInfo.workspace)/$($sourceInfo.project)/dataset/$($sourceInfo.version)"; license=$info.project.license; declaredImages=$version[0].images; projectClasses=$info.project.classes; status='metadata_verified' }
  $receiptPath=Join-Path $root "reports/$Source-download.json"
  $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
  if ($MetadataOnly) { Write-Output "$Source v$($sourceInfo.version): metadata verified"; return }
  $folder=Join-Path $root 'data/sources'
  $archive=Join-Path $folder "$Source-v$($sourceInfo.version).zip"
  if (-not (Test-Path -LiteralPath $archive)) {
    $link=$null
    for ($attempt=0; $attempt -lt 60; $attempt++) {
      $export=Get-PrivateJson "$base/$($sourceInfo.version)/yolov8?api_key=$encodedKey"
      $link=$export.export.link
      if ($link) { break }
      Start-Sleep -Seconds 5
    }
    if (-not $link) { throw 'Export not ready; rerun later.' }
    $uri=[Uri]$link
    if ($uri.Scheme -ne 'https') { throw 'Export URL must use HTTPS.' }
    Write-Output "Downloading $Source v$($sourceInfo.version) (private URL hidden)"
    $response=$client.GetAsync($uri,[System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
    try {
      if (-not $response.IsSuccessStatusCode) { throw "Export download failed (HTTP $([int]$response.StatusCode))." }
      $inputStream=$response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
      $out=[System.IO.File]::Create("$archive.part")
      try { $inputStream.CopyTo($out) } finally { $out.Dispose(); $inputStream.Dispose() }
    } finally { $response.Dispose() }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip=[System.IO.Compression.ZipFile]::OpenRead("$archive.part"); $zip.Dispose()
    Move-Item -LiteralPath "$archive.part" -Destination $archive
  }
  $receipt.status='downloaded'
  $receipt.bytes=(Get-Item -LiteralPath $archive).Length
  $receipt.sha256=(Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
  $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8
  Write-Output "${Source}: download verified"
} catch {
  # Never propagate an HTTP exception that could embed the API key or signed URL.
  $message=$_.Exception.Message
  if ($message -match 'https?://|api_key|\bkey=') { throw "Roboflow operation failed for $Source; sensitive error details suppressed." }
  throw
} finally { $client.Dispose(); $key=$null; $encodedKey=$null }
