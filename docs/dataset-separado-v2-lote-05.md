# Dataset ampliado con el lote 05

La revisión exportada pasó la validación de estructura, cajas, vínculos, identidad del revisor y hashes originales. Se aceptaron 9 de las 12 imágenes. Image_01504, Image_00924 e Image_01491 se excluyen por declaración de cobertura completa desmarcada; no se modificaron las decisiones humanas.

El lote aporta 41 cajas: 11 motos, 15 personas, 5 cabezas con casco y 10 sin casco; 15 vínculos confirmados. Las categorías visuales usadas para seleccionar candidatos no son clases nuevas ni atributos certificados.

## Nueva versión

`data/prepared/helmet-separate-v2-expanded-03`

| Partición | Imágenes | Motos | Personas | Con casco | Sin casco | Cajas |
|---|---:|---:|---:|---:|---:|---:|
| Train | 44 | 68 | 89 | 36 | 53 | 246 |
| Val | 5 | 7 | 9 | 3 | 6 | 25 |

Se conservaron las 35 imágenes anteriores y sus anotaciones. Los manifiestos y COCO de validación son idénticos byte a byte. Se verificaron los hashes de todos los archivos del lock, la carga de las 49 imágenes y la ausencia de solapamiento de train con el lote test por identificador, grupo o hash. La separación semántica depende de los grupos originales.

La preparación no ejecuta entrenamiento ni carga pesos preentrenados. El siguiente entrenamiento debe inicializar todos los pesos aleatoriamente y usar una carpeta de ejecución nueva. Para comparar duración y datos por separado, conviene registrar el paso exacto 1950 además del resultado a 150 épocas; el entrenador todavía requiere adaptar ese guardado intermedio. Con 44 imágenes y batch 2, 150 épocas implicarían 3300 actualizaciones, frente a 2700 con 35 imágenes y 1950 con 25.

## Reproducción

```powershell
.venv/Scripts/python.exe -m training.prepare_separate_labels --canonical data/annotations/train-separate-v2-05/draft.json --reviewed data/annotations/train-separate-v2-05/train-separate-v2-05-reviewed.json --split train --output data/prepared/helmet-separate-v2-staging-05
.venv/Scripts/python.exe -m training.consolidate_separate_dataset --include-batch-03 --include-batch-04 --include-batch-05 --output data/prepared/helmet-separate-v2-expanded-03
```

Los comandos rechazan sobrescribir salidas existentes. Usar destinos nuevos para otra preparación.

Evidencia: `reports/separate-v2/helmet-separate-v2-expanded-03-prepared.json` y `reports/separate-v2/expanded-03-verification.json`. Los campos conservadores `training_ready: false` del recibo de estructura y del staging no representan un fallo: esas etapas no certifican por sí solas un dataset consolidado. La versión consolidada contiene manifiestos, COCO, imágenes y lock verificados; no se ha iniciado un entrenamiento.
