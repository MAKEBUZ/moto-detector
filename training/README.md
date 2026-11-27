# Entrenamiento y evaluación de los modelos actuales

Ejecuta los comandos desde la raíz del proyecto. El [README principal](../README.md) contiene la instalación de PyTorch, la ubicación de los pesos propios y los resultados de referencia.

| Qué quieres hacer | Archivo de entrada |
|---|---|
| Repetir el ajuste del detector de ocupantes usado por v9 | [train_occupant_regions.py](train_occupant_regions.py) |
| Entrenar el especialista que busca cabezas en recortes de ocupante | [rider_crop_heads.py](rider_crop_heads.py) |
| Comparar las asociaciones del video con las marcas revisadas | [evaluate_rider_tracking.py](evaluate_rider_tracking.py) |
| Probar una foto o un video en la aplicación | [photo_app.py](photo_app.py) |

V9 es una versión del seguimiento. Su detector es `joint-occupant-negatives-11`. El ajuste parte de un checkpoint del proyecto.

## Antes de entrenar

Restaura los conjuntos preparados, los manifiestos revisados y los pesos de origen que lee el entrenador elegido. Comprueba el directorio de salida antes de ejecutar. Los entrenadores crean una carpeta nueva bajo `runs/` y se detienen si ya existe. Conserva el directorio del modelo activo cuando prepares otro experimento.

## Comprobaciones

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s training -p "test*.py"
.\.venv\Scripts\python.exe training\evaluate_rider_tracking.py --partial reports\training\tracking-occupants-v8\video.json
```

La segunda orden requiere el video local y las anotaciones de control. La referencia de v9 es 8/10 cascos de control, 9/9 de la revisión original y dos falsos ocupantes confirmados en ocho regiones negativas. [Informe de referencia](../reports/training/tracking-occupants-v9/revision.md)
