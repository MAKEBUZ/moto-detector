param([switch]$DetectorOnly)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$root=Split-Path $PSScriptRoot -Parent
$base=Join-Path $root 'data/prepared/helmet-baseline-v1'
$crops=@([System.IO.File]::ReadLines((Join-Path $base 'classifier-manifest.jsonl'))|ForEach-Object{$_|ConvertFrom-Json})
$det=@([System.IO.File]::ReadLines((Join-Path $base 'detector-manifest.jsonl'))|ForEach-Object{$_|ConvertFrom-Json})
$font=[System.Drawing.Font]::new('Arial',9)
$selected=@()
if(-not $DetectorOnly){
$canvas=[System.Drawing.Bitmap]::new(1200,540);$g=[System.Drawing.Graphics]::FromImage($canvas)
try{
 $g.Clear([System.Drawing.Color]::White);$row=0
 foreach($split in @('train','val','test')){foreach($source in @('edgevision')){
  $col=0
  foreach($class in @('con_casco','sin_casco')){
   $samples=@($crops|Where-Object {$_.split -eq $split -and $_.source -eq $source -and $_.className -eq $class}|Sort-Object id|Select-Object -First 3)
   foreach($r in $samples){
    $x=$col*200;$y=$row*180;$label="$split / $source`n$class / $($r.id.Substring(0,8))"
    $g.DrawString($label,$font,[System.Drawing.Brushes]::Black,[single]($x+4),[single]($y+3))
    $file=Join-Path $base "classifier/$split/$class/$($r.id).jpg";$img=[System.Drawing.Image]::FromFile($file)
    try{$scale=[Math]::Min(190.0/$img.Width,140.0/$img.Height);$g.DrawImage($img,[int]($x+4),[int]($y+36),[int]($img.Width*$scale),[int]($img.Height*$scale))}finally{$img.Dispose()}
    $selected+=@{task='classifier';id=$r.id;sourceImage=$r.image;split=$split;className=$class};$col++
   }
  };$row++
 }}
 $canvas.Save((Join-Path $root 'reports/preparation/classifier-sample.jpg'),[System.Drawing.Imaging.ImageFormat]::Jpeg)
}finally{$g.Dispose();$canvas.Dispose()}
}
$canvas=[System.Drawing.Bitmap]::new(1500,1050);$g=[System.Drawing.Graphics]::FromImage($canvas)
$pens=@([System.Drawing.Pen]::new([System.Drawing.Color]::Lime,2),[System.Drawing.Pen]::new([System.Drawing.Color]::Cyan,2),[System.Drawing.Pen]::new([System.Drawing.Color]::OrangeRed,2))
try{
 $g.Clear([System.Drawing.Color]::White);$row=0
 foreach($split in @('train','val','test')){
  $col=0
  foreach($r in @($det|Where-Object split -eq $split|Sort-Object id|Select-Object -First 3)){
   $x=$col*500;$y=$row*350;$g.DrawString("$split / $($r.id)",$font,[System.Drawing.Brushes]::Black,[single]($x+4),[single]($y+3))
   $img=[System.Drawing.Image]::FromFile((Join-Path $base "detector/images/$split/$($r.id).jpg"))
   try{
    $scale=[Math]::Min(490.0/$img.Width,315.0/$img.Height);$g.DrawImage($img,[int]($x+4),[int]($y+25),[int]($img.Width*$scale),[int]($img.Height*$scale))
    foreach($a in $r.annotations){$b=$a.box;$g.DrawRectangle($pens[$a.classId],[single]($x+4+$b[0]*$scale),[single]($y+25+$b[1]*$scale),[single](($b[2]-$b[0])*$scale),[single](($b[3]-$b[1])*$scale))}
   }finally{$img.Dispose()}
   $selected+=@{task='detector';id=$r.id;sourceImage=$r.image;split=$split};$col++
  };$row++
 }
 $canvas.Save((Join-Path $root 'reports/preparation/detector-sample.jpg'),[System.Drawing.Imaging.ImageFormat]::Jpeg)
 $selected|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $root 'reports/preparation/visual-sample.json') -Encoding utf8
}finally{foreach($p in $pens){$p.Dispose()};$g.Dispose();$canvas.Dispose();$font.Dispose()}
