# Dataset separado v2 preparado

Ubicación: `data/prepared/helmet-separate-v2/`.

| Partición | Imágenes | Cajas | Vínculos |
|---|---:|---:|---:|
| Entrenamiento | 14 | 69 | 25 |
| Validación | 5 | 25 | 9 |

| Clase | ID manifiesto | ID COCO / salida del detector | Train | Val |
|---|---:|---:|---:|---:|
| motocicleta | 0 | 1 | 19 | 7 |
| persona_ocupante | 1 | 2 | 25 | 9 |
| cabeza_con_casco | 2 | 3 | 15 | 3 |
| cabeza_sin_casco | 3 | 4 | 10 | 6 |

La salida 0 del detector se reservará para fondo. `persona_ocupante` no es una clase general de peatones. Las etiquetas de moto ya no significan moto+ocupante.

## Archivos

- `images/train/` y `images/val/`: copias verificadas de los originales aceptados.
- `train-manifest.jsonl` y `val-manifest.jsonl`: cajas xyxy en píxeles originales, IDs de entidades, vínculos, grupos y procedencia. El campo `image` es relativo a la raíz del dataset; `source_image` conserva la ruta de origen.
- `train-coco.json` y `val-coco.json`: detección en formato COCO, cajas xywh; `file_name` relativo a la raíz. Los vínculos se conservan en los manifiestos, no en COCO.
- `dataset.json`: estadísticas, exclusiones y hashes de los archivos de revisión y preparación.
- `dataset.lock.json`: hashes de todos los archivos del dataset salvo el propio lock.

## Exclusiones conservadas

Primer lote train: Image_01416 e Image_00419 por cobertura no confirmada.

Segundo lote train: Image_00265, Image_01871 e Image_00535 por entidades inciertas; Image_00101 por cobertura no confirmada.

Validación: Image_01819, Image_01327 e Image_00571 por cobertura no confirmada.

Estas decisiones no se modificaron ni se interpretaron como errores que el asistente pudiera confirmar. Se excluyen imágenes completas para no convertir objetos omitidos en fondo.

## Comprobaciones

Las dos revisiones nuevas y la revisión anterior pasaron validación estructural, referencias, dimensiones y hashes. Los manifiestos intermedios se compararon nuevamente contra las revisiones humanas. Se verificaron las particiones y grupos originales, ausencia de hashes/IDs duplicados y ausencia de grupos compartidos entre train y val. La agrupación original no certifica independencia absoluta de escenas. No se usó test ni el antiguo piloto de siete imágenes.

Se comprobaron las imágenes copiadas, las referencias COCO y los 24 archivos registrados en el lock. Pasaron 23 pruebas unitarias. El constructor reproducible es `python -m training.consolidate_separate_dataset`; rechaza sobrescribir una carpeta existente.

## Próximo entrenamiento

Las anotaciones están preparadas, pero el código de entrenamiento anterior sigue esperando tres clases y otro formato de dataset. Hay que adaptar cargador, salida de cinco categorías (fondo + cuatro clases), métricas e inferencia. La inicialización debe ser completamente aleatoria, sin cargar checkpoints anteriores ni pesos externos.

No se inició entrenamiento. Este dataset pequeño sirve para una primera prueba del flujo, no para afirmar robustez o habilitar conteo en producción. La validación tiene cinco imágenes y solo tres cajas con casco; sus métricas tendrán gran variabilidad.
