param([string]$SourceRoot = 'data/sources/originals/edgevision-v1')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$project = Split-Path $PSScriptRoot -Parent
$base = Join-Path $project $SourceRoot
if (-not (Test-Path -LiteralPath $base)) { throw 'Extract the original archive first.' }
$reportDir = Join-Path $project 'reports'
$manifestPath = Join-Path $reportDir 'edgevision-manifest.jsonl'
$errors = [System.Collections.Generic.List[object]]::new()
$all = @(Get-ChildItem -LiteralPath $base -Recurse -File | Where-Object { $_.FullName -notmatch '__MACOSX' })
$images = @($all | Where-Object Extension -Match '^\.(jpg|jpeg|png)$' | Sort-Object FullName)
$xmls = @($all | Where-Object Extension -EQ '.xml')
$byName = @{}
$imageStems = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($file in $images) {
  [void]$imageStems.Add($file.BaseName)
  $key = $file.Name.ToLowerInvariant()
  if ($byName.ContainsKey($key)) { $errors.Add(@{ type='ambiguous_image_name'; name=$key }); continue }
  $byName[$key] = $file
}
$byXmlName = @{}
foreach ($file in $xmls) {
  $key = $file.BaseName.ToLowerInvariant()
  if ($byXmlName.ContainsKey($key)) { $errors.Add(@{ type='ambiguous_xml_name'; name=$key }); continue }
  $byXmlName[$key] = $file
}
$unified = @{}
foreach ($split in @('train','val','test')) {
  $dir = Join-Path $project "data/sources/DataSetsIA/datasets_casco/unificado/labels/$split"
  foreach ($file in Get-ChildItem -LiteralPath $dir -Filter 'edgevision__*.txt') {
    $key = $file.BaseName.Substring('edgevision__'.Length).ToLowerInvariant()
    $unified[$key] = @{ split=$split; file=$file }
  }
}
$classMap = @{ BikeWithRider=0; Helmet=1; NoHelmet=3 }
$counts = @{ BikeWithRider=0; Helmet=0; NoHelmet=0 }
$decodeCache=@{}
if(Test-Path -LiteralPath $manifestPath){
  foreach($line in [System.IO.File]::ReadLines($manifestPath)){
    try{$cached=$line|ConvertFrom-Json;if($cached.decoded){$decodeCache[$cached.sha256]=$cached}}catch{}
  }
}
$writer = [System.IO.StreamWriter]::new($manifestPath, $false, [System.Text.UTF8Encoding]::new($false))
$processed = 0
$decoded = 0
$matched = 0
$withXml = 0
function Rel([string]$p) { return [System.IO.Path]::GetRelativePath($project, $p).Replace('\','/') }
function Number([string]$v) { return [double]::Parse($v, [Globalization.CultureInfo]::InvariantCulture) }
try {
  foreach ($file in $images) {
    $key = $file.BaseName.ToLowerInvariant()
    $record = [ordered]@{ source='edgevision'; version=1; image=(Rel $file.FullName); sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant(); width=$null; height=$null; decoded=$false; dhash=$null; originalXml=$null; upstreamSplit=$null; groupId=$null; groupStatus='scene_unknown'; annotations=@(); unifiedLabelsMatch=$false }
    $img = $null; $full = $null; $graphics = $null; $thumb=$null; $tg=$null
    if($decodeCache.ContainsKey($record.sha256)){
      $cached=$decodeCache[$record.sha256];$record.width=$cached.width;$record.height=$cached.height;$record.dhash=$cached.dhash;$record.decoded=$true;$decoded++
    }else{
    try {
      $img = [System.Drawing.Image]::FromFile($file.FullName)
      $record.width = $img.Width; $record.height = $img.Height
      $full = [System.Drawing.Bitmap]::new($img.Width, $img.Height)
      $graphics = [System.Drawing.Graphics]::FromImage($full)
      $graphics.DrawImage($img, 0, 0, $img.Width, $img.Height)
      $record.decoded=$true; $decoded++
      $thumb = [System.Drawing.Bitmap]::new(9,8)
      $tg = [System.Drawing.Graphics]::FromImage($thumb)
      $tg.DrawImage($full,0,0,9,8)
      $bits=''
      for ($y=0; $y -lt 8; $y++) { for ($x=0; $x -lt 8; $x++) {
        $a=$thumb.GetPixel($x,$y); $b=$thumb.GetPixel($x+1,$y)
        $la=299*$a.R+587*$a.G+114*$a.B; $lb=299*$b.R+587*$b.G+114*$b.B
        $bits += $(if ($la -gt $lb) {'1'} else {'0'})
      } }
      $record.dhash=[Convert]::ToUInt64($bits,2).ToString('x16')
    } catch { $errors.Add(@{ type='image_decode_failed'; file=(Rel $file.FullName); message=$_.Exception.Message }) }
    finally {
      foreach ($obj in @($tg,$thumb,$graphics,$full,$img)) { if ($null -ne $obj) { $obj.Dispose() } }
    }
    }
    if ($byXmlName.ContainsKey($key)) {
      $xmlFile=$byXmlName[$key]; $record.originalXml=Rel $xmlFile.FullName
      try {
        $settings=[System.Xml.XmlReaderSettings]::new(); $settings.DtdProcessing=[System.Xml.DtdProcessing]::Prohibit; $settings.XmlResolver=$null
        $reader=[System.Xml.XmlReader]::Create($xmlFile.FullName,$settings)
        $xml=[System.Xml.XmlDocument]::new(); $xml.XmlResolver=$null
        try { $xml.Load($reader) } finally { $reader.Dispose() }
        $withXml++
        $w=Number $xml.annotation.size.width; $h=Number $xml.annotation.size.height
        if ($w -ne $record.width -or $h -ne $record.height) { $errors.Add(@{type='xml_image_size_mismatch'; file=Rel $xmlFile.FullName}) }
        if ([System.IO.Path]::GetFileName([string]$xml.annotation.filename) -ne $file.Name) { $errors.Add(@{type='xml_filename_mismatch'; file=Rel $xmlFile.FullName}) }
        $expected=[System.Collections.Generic.List[object]]::new()
        foreach ($obj in @($xml.annotation.object)) {
          if ($null -eq $obj) { continue }
          $name=[string]$obj.name
          if (-not $classMap.ContainsKey($name)) { $errors.Add(@{type='unknown_class'; class=$name; file=Rel $xmlFile.FullName}); continue }
          $counts[$name]++
          $x0=Number $obj.bndbox.xmin; $y0=Number $obj.bndbox.ymin; $x1=Number $obj.bndbox.xmax; $y1=Number $obj.bndbox.ymax
          $record.annotations += @{class=$name; bbox=@($x0,$y0,$x1,$y1)}
          if ($x0 -lt 0 -or $y0 -lt 0 -or $x1 -gt $w -or $y1 -gt $h -or $x1 -le $x0 -or $y1 -le $y0) { $errors.Add(@{type='invalid_original_box'; file=Rel $xmlFile.FullName}) }
          $x0=[Math]::Max(0,$x0); $y0=[Math]::Max(0,$y0); $x1=[Math]::Min($w,$x1); $y1=[Math]::Min($h,$y1)
          if ($w -gt 0 -and $h -gt 0 -and $x1 -gt $x0 -and $y1 -gt $y0) { $expected.Add(@([double]$classMap[$name],(($x0+$x1)/2/$w),(($y0+$y1)/2/$h),(($x1-$x0)/$w),(($y1-$y0)/$h))) }
        }
        if ($unified.ContainsKey($key)) {
          $u=$unified[$key]; $record.upstreamSplit=$u.split
          $actual=@(Get-Content -LiteralPath $u.file.FullName | Where-Object { $_.Trim() })
          $same=$actual.Count -eq $expected.Count
          if ($same) { for ($i=0; $i -lt $actual.Count; $i++) {
            $v=@($actual[$i] -split '\s+' | ForEach-Object { Number $_ })
            if ($v.Count -ne 5) { $same=$false; break }
            for ($j=0; $j -lt 5; $j++) { if ([Math]::Abs($v[$j]-$expected[$i][$j]) -gt 0.000002) { $same=$false } }
          } }
          $record.unifiedLabelsMatch=$same
          if ($same) { $matched++ } else { $errors.Add(@{type='unified_label_mismatch'; file=Rel $u.file.FullName}) }
        } else { $errors.Add(@{type='missing_unified_label'; image=Rel $file.FullName}) }
      } catch { $errors.Add(@{type='xml_processing_failed'; file=Rel $xmlFile.FullName; message=$_.Exception.Message}) }
    } else { $errors.Add(@{type='missing_xml'; image=Rel $file.FullName}) }
    $writer.WriteLine(($record | ConvertTo-Json -Depth 8 -Compress))
    $processed++
    if ($processed % 250 -eq 0) { Write-Output "Validated $processed / $($images.Count) images" }
  }
} finally { $writer.Dispose() }
foreach ($key in $unified.Keys) { if (-not $imageStems.Contains($key)) { $errors.Add(@{type='label_without_original'; file=Rel $unified[$key].file.FullName}) } }
foreach ($key in $byXmlName.Keys) { if (-not $imageStems.Contains($key)) { $errors.Add(@{type='xml_without_image'; file=Rel $byXmlName[$key].FullName}) } }
$summary=[ordered]@{ source='edgevision'; version=1; generatedAtUtc=[DateTime]::UtcNow.ToString('o'); images=$images.Count; xmlFiles=$xmls.Count; decodedImages=$decoded; matchedXml=$withXml; unifiedLabelFiles=$unified.Count; matchingUnifiedLabels=$matched; boxesByOriginalClass=$counts; errors=@($errors.ToArray()); scope='Full-resolution rendering, SHA256, dHash candidates, XML dimensions/boxes and comparison with unified labels. Scene membership remains unknown.' }
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $reportDir 'edgevision-originals-audit.json') -Encoding utf8
[pscustomobject]$summary | Select-Object images,xmlFiles,decodedImages,matchingUnifiedLabels | Format-List
Write-Output "Issues: $($errors.Count)"
