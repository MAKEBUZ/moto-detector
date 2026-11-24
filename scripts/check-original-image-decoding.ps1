param([string]$Manifest = 'reports/roboflow-manifest.jsonl', [string]$Output = 'reports/roboflow-image-decoding.jsonl')
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$root=Split-Path $PSScriptRoot -Parent
$writer=[System.IO.StreamWriter]::new((Join-Path $root $Output),$false,[System.Text.UTF8Encoding]::new($false))
$n=0; $failed=0
try {
 foreach($line in [System.IO.File]::ReadLines((Join-Path $root $Manifest))) {
  $r=$line | ConvertFrom-Json
  $row=[ordered]@{image=$r.image;decoded=$false;width=$null;height=$null;dhash=$null}
  $img=$null;$full=$null;$g=$null;$thumb=$null;$tg=$null
  try {
   $img=[System.Drawing.Image]::FromFile((Join-Path $root $r.image))
   $full=[System.Drawing.Bitmap]::new($img.Width,$img.Height)
   $g=[System.Drawing.Graphics]::FromImage($full);$g.DrawImage($img,0,0,$img.Width,$img.Height)
   $row.width=$img.Width;$row.height=$img.Height
   $thumb=[System.Drawing.Bitmap]::new(9,8);$tg=[System.Drawing.Graphics]::FromImage($thumb);$tg.DrawImage($full,0,0,9,8)
   $bits=''
   for($y=0;$y -lt 8;$y++){for($x=0;$x -lt 8;$x++){
    $a=$thumb.GetPixel($x,$y);$b=$thumb.GetPixel($x+1,$y)
    $bits += $(if((299*$a.R+587*$a.G+114*$a.B) -gt (299*$b.R+587*$b.G+114*$b.B)){'1'}else{'0'})
   }}
   $row.dhash=[Convert]::ToUInt64($bits,2).ToString('x16');$row.decoded=$true
  } catch { $row.error=$_.Exception.Message; $failed++ }
  finally {foreach($o in @($tg,$thumb,$g,$full,$img)){if($null -ne $o){$o.Dispose()}}}
  $writer.WriteLine(($row|ConvertTo-Json -Compress));$n++
  if($n%1000 -eq 0){Write-Output "Decoded $n images; failures=$failed"}
 }
}finally{$writer.Dispose()}
Write-Output "Completed: $n images; failures=$failed"
