# Entrenamiento desde cero

Se usa PyTorch como biblioteca, sin pesos preentrenados. `models.py` construye una CNN y un Faster R-CNN con backbone convolucional propio inicializados aleatoriamente. No se utiliza ResNet preentrenada, COCO, ImageNet ni checkpoints externos.

El detector usa tres clases más fondo; las etiquetas YOLO 0/1/2 se convierten a 1/2/3 durante el entrenamiento. Su resolución de trabajo es de 384 a 640 píxeles, para una primera prueba con recursos locales. Es una arquitectura inicial; todavía no está optimizada para tiempo real ni objetos muy pequeños.

El clasificador recibe recortes de 96 × 96 con relación de aspecto conservada mediante relleno. Los aumentos son brillo y espejo horizontal, solo en entrenamiento. Usa pesos de clase calculados únicamente a partir del conjunto de entrenamiento.

## Entorno

Python local en `.venv`. Las bibliotecas CUDA se descargan del índice oficial de PyTorch; no son pesos de modelos. No se necesita instalar globalmente un CUDA Toolkit para utilizar estas ruedas.

```powershell
./.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
./.venv/Scripts/python.exe training/test_metrics.py
```

Cada ejecución guarda versiones exactas en `runs/<nombre>/requirements.lock.txt`.

La instalación utilizada también está fijada en `training/requirements-cu128.lock.txt`. Para repetirla: `./.venv/Scripts/python.exe -m pip install -r training/requirements-cu128.lock.txt --index-url https://download.pytorch.org/whl/cu128`.

## Ejecuciones

Desde la raíz del proyecto:

```powershell
# Pruebas pequeñas; no son resultados representativos.
./.venv/Scripts/python.exe training/train.py --task classifier --epochs 2 --limit-train 128 --limit-val 64 --run classifier-smoke
./.venv/Scripts/python.exe training/train.py --task detector --epochs 2 --limit-train 16 --limit-val 8 --run detector-smoke

# Primer entrenamiento con todos los ejemplos de train y val.
./.venv/Scripts/python.exe training/train.py --task classifier --epochs 15 --run classifier-baseline
./.venv/Scripts/python.exe training/train.py --task detector --epochs 5 --run detector-baseline
```

Los nombres de ejecución son únicos y no se sobrescriben. No hay reanudación implícita ni parámetro para cargar pesos externos. El bloqueo de las funciones de descarga de torch.hub es una comprobación adicional; la garantía principal es que los constructores no solicitan pesos y el entrenador no llama a `torch.load`.

Cada carpeta contiene configuración, `history.json`, `result.json`, `best.pt` y `last.pt`. Los checkpoints generados son pesos propios aprendidos aquí. `best.pt` se elige por macro-F1 de validación para clasificación y por mAP@0.50 para detección. `last.pt` permite inspeccionar el estado de la última época; se conserva el optimizador, pero este entrenador todavía no implementa reanudación.

## Métricas y reproducibilidad

- Clasificación: matriz de confusión (filas reales, columnas predichas), precisión, recall y F1 de ambas clases, macro-F1 y accuracy.
- Detección: AP interpolada en 101 puntos, IoU 0,50, máximo 100 predicciones por imagen; se informa por clase y su media. No equivale a COCO mAP@0.50:0.95.
- No se utiliza `test`: se reserva hasta fijar el experimento y los umbrales.
- Se registran semillas, versiones, huella de los pesos iniciales y finales y huellas del dataset. Una semilla no garantiza igualdad bit a bit entre GPUs/versiones distintas.
- Las particiones son un baseline interno con agrupación heurística. El desempeño en cámara real necesita una prueba externa.

Los resultados de un entrenamiento corto no garantizan uso fiable, conteo de pasajeros ni casco bien abrochado. Véase `docs/dataset-preparado.md` para el alcance real de las etiquetas.

## Inferencia con los checkpoints propios

```powershell
./.venv/Scripts/python.exe training/predict.py --run classifier-baseline --image "ruta/al/recorte-de-cabeza.jpg" --output reports/training/mi-clasificacion
./.venv/Scripts/python.exe training/predict.py --run detector-baseline --image "ruta/a/la/escena.jpg" --output reports/training/mi-deteccion --threshold 0.5
```

Sustituir las rutas de ejemplo por archivos existentes. Se generan resultados JSON y, para detección, una imagen anotada. El clasificador requiere un recorte de cabeza; no usar una escena completa como si fuera una cabeza. El umbral 0,5 es una opción inicial no calibrada para multas ni alertas. `predict.py` carga únicamente el checkpoint propio de la carpeta `runs` indicada; entrenar no carga ningún checkpoint.

La inferencia del detector incluye ahora `association`: propuestas geométricas cabeza–moto, ambigüedades y conflictos. No son ocupantes confirmados y no habilitan conteo. La revisión y sus límites están documentados en `reports/error-review/README.md`.

Referencias de las APIs: [instalación de PyTorch](https://pytorch.org/get-started/locally/) y el código instalado de `torchvision.models.detection.faster_rcnn.FasterRCNN`.

## Reserva experimental de positivos RPN

`training/train.py --rpn-object-reserve` activa la reserva de un positivo por objeto que disponga de anclas asignadas positivas. Es opcional y solo se permite en el detector de cuatro clases. Se mantiene el tamaño del muestreo y su proporción positiva, y se conservan las negativas del sorteo original. Se reparan omisiones intercambiando positivos redundantes; si no alcanza el presupuesto, se detiene explícitamente. Las reparaciones usan un RNG separado con semilla de ejecución +1000000.

`rpn-selections.jsonl` registra todos los pasos e imágenes, identificadores de objetos, clases, positivos disponibles, selección habitual previa a reparar, selección final e índices de anclas. No registra coordenadas completas ni las imágenes aumentadas. Los checkpoints guardan pesos con el mismo esquema; la inferencia no necesita el sampler y no cambia por instalarlo. No hay reanudación implícita.

Experimento reproducible: `training/run_fixed_updates.py --rpn-object-reserve`. Comparación: `training/report_object_reserve.py`. Requieren destinos nuevos si ya existen resultados. La reserva no se adoptó como configuración predeterminada: su mejora no fue consistente en las dos semillas y datasets evaluados. Ver `reports/training/object-reserve-01/README.md`.
