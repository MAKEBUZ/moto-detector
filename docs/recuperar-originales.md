# Recuperación reproducible de originales

Requiere Windows, PowerShell 7, Git y Node.js. No requiere Python ni modelos de IA. Los archivos pesados se guardan en `data/sources/`, excluido de Git. Las versiones están fijadas en `data/sources.lock.json`.

## Roboflow

Crear `.env` en la raíz con `ROBOFLOW_API_KEY=` seguido de la clave privada. No compartir la clave ni incorporar `.env` al repositorio. El descargador lee el archivo localmente y evita registrar URLs privadas.

```powershell
./scripts/download-roboflow.ps1 -Source motorcycle_helmet_od
./scripts/download-roboflow.ps1 -Source mapua_helmet
./scripts/download-roboflow.ps1 -Source ldrago_helmet
./scripts/download-roboflow.ps1 -Source proper_helmet_usage
./scripts/download-roboflow.ps1 -Source helmet_violation
```

Los recibos `reports/*-download.json` guardan versión, licencia declarada por la API, tamaño y SHA-256 local; no incluyen claves. Una licencia declarada por el publicador no es una auditoría de la procedencia de cada imagen.

## EdgeVision

```powershell
./scripts/download-public-source.ps1 -Source edgevision
```

Si la conexión se atasca, detener ese proceso antes de utilizar `./scripts/download-edgevision-ranges.ps1`. Este último está fijado al tamaño comprobado de EdgeVision v1 y descarga fragmentos de 4 MB, conservando un prefijo parcial. No ejecutar ambos descargadores a la vez. Los fragmentos se ensamblan en orden y se verifica el tamaño y la estructura ZIP; la decodificación de imágenes se verifica después. El hash local sirve para identificar la descarga, no reemplaza una firma o checksum oficial.

## Extraer y auditar

```powershell
./scripts/extract-source.ps1
$lock = Get-Content data/sources.lock.json -Raw | ConvertFrom-Json
foreach ($s in $lock.sources | Where-Object provider -eq 'roboflow') {
  ./scripts/extract-source.ps1 -Archive "data/sources/$($s.id)-v$($s.version).zip" -Destination "data/sources/originals/$($s.id)-v$($s.version)"
}
node scripts/audit-roboflow-originals.mjs
./scripts/check-original-image-decoding.ps1
./scripts/audit-edgevision-originals.ps1
node scripts/find-similar-originals.mjs
./scripts/preview-originals.ps1
```

Los manifiestos JSONL contienen una fila por imagen. Conservan fuente, versión, SHA-256, anotaciones originales y partición del agregador; no cambian esa partición. `scene_unknown` significa que aún no conocemos la escena: no debe interpretarse como una escena distinta por imagen.

La comparación contra el agregador reproduce su mapeo para comprobar correspondencia. Que coincida no valida la semántica de las clases, las cajas de cabeza/persona ni la exhaustividad de las etiquetas.

Las similitudes dHash son candidatos para revisión humana. Su umbral no demuestra identidad; los grupos conectados pueden incluir imágenes distintas. No se eliminan archivos ni se generan particiones finales automáticamente.

## Fuentes y atribución

- **EdgeVision v1**: Kandarp Gajjar, Nancy Patel, Ridham Patel, Krutika Patel, Chrisha Dabhi, Himani Trivedi, Mahesh Goyani y Chintan Bhatt. DOI [10.17632/j82bnw7gsr.1](https://data.mendeley.com/datasets/j82bnw7gsr/1). CC BY 4.0, comprobada en la página oficial el 23/11/2026. Se conservaron originales sin modificar.
- **Motorcycle Helmet Object Detection v1**: [projectt](https://universe.roboflow.com/projectt/motorcycle-helmet-object-detection/dataset/1).
- **Motorcycle Helmet Detection Dataset v8**: [Mapua University](https://universe.roboflow.com/mapua-university/motorcycle-helmet-detection-dataset/dataset/8).
- **Helmet v10**: [ldrago](https://universe.roboflow.com/ldrago/helmet-elr7r/dataset/10).
- **Proper Helmet Usage v6**: [duusuuuu](https://universe.roboflow.com/duusuuuu/proper-helmet-usage/dataset/6).
- **Helmet Violation Detection v1**: [imageprocessingworkspace](https://universe.roboflow.com/imageprocessingworkspace/helmet-violation-detetction-ihuwu/dataset/1).

Las cinco fuentes Roboflow declaran CC BY 4.0 en la API; los exports incluyen los archivos README de atribución y se conservan junto con las imágenes.

El dataset de [videos de Zacatecas v1](https://data.mendeley.com/datasets/bmy35m25pw/1), de Jaime Mercado Reyna y colaboradores, también declara CC BY 4.0. Su descarga se reserva para la fase de video: sus 32 grabaciones tienen una perspectiva orientada al conductor y no sustituyen una prueba de cámara vial externa.
