param([string]$DatasetRoot = 'data/sources/DataSetsIA', [string]$Output = 'reports/crop-sample.jpg')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$base = Join-Path $DatasetRoot 'datasets_casco/unificado/recortes/train'
$sources = @('edgevision','motorcycle_helmet_od','mapua_helmet','ldrago_helmet','proper_helmet_usage','helmet_violation')
$classes = @('casco_correcto','casco_mal_puesto','sin_casco')
$canvas = [System.Drawing.Bitmap]::new(960, 1080)
$graphics = [System.Drawing.Graphics]::FromImage($canvas)
$font = [System.Drawing.Font]::new('Arial', 10)
try {
  $graphics.Clear([System.Drawing.Color]::White)
  $selected = @()
  for ($row = 0; $row -lt $sources.Count; $row++) {
    for ($col = 0; $col -lt $classes.Count; $col++) {
      $source = $sources[$row]
      $class = $classes[$col]
      $x = $col * 320
      $y = $row * 180
      $graphics.DrawString("$source / $class", $font, [System.Drawing.Brushes]::Black, $x + 4, $y + 4)
      $file = Get-ChildItem -LiteralPath (Join-Path $base $class) -Filter "$($source)__*.jpg" | Sort-Object Name | Select-Object -First 1
      if ($null -eq $file) { continue }
      $selected += $file.FullName
      $img = [System.Drawing.Image]::FromFile($file.FullName)
      try {
        $scale = [Math]::Min(300.0 / $img.Width, 146.0 / $img.Height)
        $graphics.DrawImage($img, [int]($x + 8), [int]($y + 28), [int]($img.Width * $scale), [int]($img.Height * $scale))
      } finally { $img.Dispose() }
    }
  }
  $target = [System.IO.Path]::GetFullPath($Output)
  [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($target)) | Out-Null
  $canvas.Save($target, [System.Drawing.Imaging.ImageFormat]::Jpeg)
  $selected | Set-Content -LiteralPath ([System.IO.Path]::ChangeExtension($target, '.txt'))
  Write-Output $target
} finally { $font.Dispose(); $graphics.Dispose(); $canvas.Dispose() }
