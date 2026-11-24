$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$root=Split-Path $PSScriptRoot -Parent
$records=@(Get-Content -LiteralPath (Join-Path $root 'reports/roboflow-manifest.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
if(Test-Path -LiteralPath (Join-Path $root 'reports/edgevision-manifest.jsonl')){ $records+=@(Get-Content -LiteralPath (Join-Path $root 'reports/edgevision-manifest.jsonl') | ForEach-Object { $_ | ConvertFrom-Json }) }
$sources=@('edgevision','motorcycle_helmet_od','mapua_helmet','ldrago_helmet','proper_helmet_usage','helmet_violation')
$canvas=[System.Drawing.Bitmap]::new(1500,1200);$g=[System.Drawing.Graphics]::FromImage($canvas);$font=[System.Drawing.Font]::new('Arial',11);$pen=[System.Drawing.Pen]::new([System.Drawing.Color]::Lime,2)
$selected=@()
try{
 $g.Clear([System.Drawing.Color]::White)
 for($i=0;$i -lt $sources.Count;$i++){
  $source=$sources[$i];$r=@($records|Where-Object source -eq $source)
  $samples=@(($r|Where-Object {$_.annotations.Count -gt 0}|Select-Object -First 1),($r|Where-Object {$_.annotations.Count -eq 0}|Select-Object -First 1))
  for($j=0;$j -lt 2;$j++){
   $x=($i%3)*500;$y=[Math]::Floor($i/3)*600+$j*300
   $label=if($j -eq 0){'con anotaciones'}else{'sin anotaciones'}
   $g.DrawString("$source / $label",$font,[System.Drawing.Brushes]::Black,[single]($x+5),[single]($y+5))
   $s=$samples[$j];if($null -eq $s){continue};$selected+=$s.image
   $img=[System.Drawing.Image]::FromFile((Join-Path $root $s.image))
   try{
    $scale=[Math]::Min(480.0/$img.Width,260.0/$img.Height)
    $g.DrawImage($img,[int]($x+5),[int]($y+30),[int]($img.Width*$scale),[int]($img.Height*$scale))
    foreach($a in $s.annotations){
     if($a.bbox){$b=$a.bbox;$bx=$b[0];$by=$b[1];$bw=$b[2]-$b[0];$bh=$b[3]-$b[1]}
     elseif($a.coordinates.Count -eq 4){$b=$a.coordinates;$bx=($b[0]-$b[2]/2)*$img.Width;$by=($b[1]-$b[3]/2)*$img.Height;$bw=$b[2]*$img.Width;$bh=$b[3]*$img.Height}
     elseif($a.coordinates.Count -ge 6){
      $xs=@();$ys=@();for($k=0;$k -lt $a.coordinates.Count;$k+=2){$xs+=$a.coordinates[$k];$ys+=$a.coordinates[$k+1]}
      $bx=($xs|Measure-Object -Minimum).Minimum*$img.Width;$by=($ys|Measure-Object -Minimum).Minimum*$img.Height
      $bw=($xs|Measure-Object -Maximum).Maximum*$img.Width-$bx;$bh=($ys|Measure-Object -Maximum).Maximum*$img.Height-$by
     }else{continue}
     $g.DrawRectangle($pen,[single]($x+5+$bx*$scale),[single]($y+30+$by*$scale),[single]($bw*$scale),[single]($bh*$scale))
    }
   }finally{$img.Dispose()}
  }
 }
 $canvas.Save((Join-Path $root 'reports/originals-sample.jpg'),[System.Drawing.Imaging.ImageFormat]::Jpeg)
 $selected|Set-Content -LiteralPath (Join-Path $root 'reports/originals-sample.txt')
}finally{$pen.Dispose();$font.Dispose();$g.Dispose();$canvas.Dispose()}
