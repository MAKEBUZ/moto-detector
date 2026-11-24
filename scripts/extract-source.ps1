param([string]$Archive = 'data/sources/edgevision-v1.zip', [string]$Destination = 'data/sources/originals/edgevision-v1')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$workspace = [System.IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))
$dest = [System.IO.Path]::GetFullPath((Join-Path $workspace $Destination))
if (-not $dest.StartsWith(($workspace + [System.IO.Path]::DirectorySeparatorChar), [StringComparison]::OrdinalIgnoreCase)) { throw 'Destination must be inside the project.' }
function Expand-SafeZip([string]$ZipPath, [string]$Target) {
  [System.IO.Directory]::CreateDirectory($Target) | Out-Null
  $prefix = [System.IO.Path]::GetFullPath($Target).TrimEnd('\','/') + [System.IO.Path]::DirectorySeparatorChar
  $zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
  try {
    foreach ($entry in $zip.Entries) {
      $out = [System.IO.Path]::GetFullPath((Join-Path $Target $entry.FullName))
      if (-not $out.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe ZIP entry: $($entry.FullName)" }
      if ($entry.FullName.Contains(':')) { throw 'Alternate data stream or absolute path in ZIP.' }
    }
    foreach ($entry in $zip.Entries) {
      $out = [System.IO.Path]::GetFullPath((Join-Path $Target $entry.FullName))
      if (-not $entry.Name) { [System.IO.Directory]::CreateDirectory($out) | Out-Null; continue }
      [System.IO.Directory]::CreateDirectory((Split-Path $out -Parent)) | Out-Null
      # Re-extraction restores known originals from the same archive, without deleting other files.
      [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $out, $true)
    }
  } finally { $zip.Dispose() }
}
Expand-SafeZip ([System.IO.Path]::GetFullPath((Join-Path $workspace $Archive))) $dest
$nested = @(Get-ChildItem -LiteralPath $dest -Recurse -File -Filter '*.zip')
foreach ($file in $nested) { Expand-SafeZip $file.FullName (Join-Path $file.DirectoryName $file.BaseName) }
Get-ChildItem -LiteralPath $dest -Recurse -File | Group-Object Extension | Select-Object Name,Count
