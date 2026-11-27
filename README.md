# Detector de casco en motociclistas

La aplicación de escritorio detecta motos y ocupantes, asocia cada cabeza a su ocupante y guarda el resultado como **con casco**, **sin casco** o **indeterminado**. La rama `main` contiene la configuración de referencia con pesos propios y el seguimiento v9 para video panorámico.

## Cómo ejecutar la app

Desde PowerShell, en la raíz del proyecto ya instalado:

```powershell
.\.venv\Scripts\python.exe training\photo_app.py
```

Para instalarla en otro equipo Windows con Git y Python 3.14 de 64 bits:

```powershell
git clone --branch main https://github.com/MAKEBUZ/moto-detector.git
cd moto-detector
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r training\requirements-cu128.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install opencv-python-headless==5.1.0.95
.\.venv\Scripts\python.exe training\photo_app.py
```

El entorno de referencia usa Tkinter, PyTorch con CUDA 12.8 y una RTX 3060. El cargador selecciona CPU cuando CUDA no está disponible. Los pesos necesarios vienen en el clon.

| Acción | Resultado |
|---|---|
| **Cargar foto** | Muestra cajas, asociaciones y conteos. Guarda la foto anotada y su JSON en `reports/fotos/`. |
| **Cargar video** | Analiza el archivo por fotogramas y cuenta trayectorias. Al terminar guarda `video.json` y `anotado.avi` en `reports/videos/<id>/`. |
| **Volver a cargar el video original** | Si coinciden el contenido, el modelo y la configuración, reproduce la copia anotada a los FPS originales. La copia usa la resolución de vista previa y se guarda sin audio. |
| **Usar cámara** | Procesa la cámara local. **Detener y guardar** cierra la captura y guarda el conteo observado. |
| **Consultar estadísticas** | Admite consultas como «¿Cuántos ocupantes con casco hubo hoy en videos?». Suma los informes guardados. Reproducir un video en caché conserva el informe existente. |

## Resultados medidos

El detector de video obtuvo **80,43 % de mAP50** en **5 fotos de validación con 25 cajas**. AP50 mide detección con solapamiento IoU ≥ 0,50. La precisión y el recobrado de la tabla corresponden al umbral de confianza 0,50. La clase sin casco tuvo el menor recobrado.

| Clase | Cajas anotadas | AP50 | Precisión | Recobrado |
|---|---:|---:|---:|---:|
| Motocicleta | 7 | 80,53 % | 71,43 % | 71,43 % |
| Ocupante | 9 | 78,01 % | 87,50 % | 77,78 % |
| Cabeza con casco | 3 | 100,00 % | 75,00 % | 100,00 % |
| Cabeza sin casco | 6 | 63,17 % | 75,00 % | 50,00 % |

| Video de prueba | Fotogramas | Motos | Ocupantes | Con casco | Sin casco | Indeterminados |
|---|---:|---:|---:|---:|---:|---:|
| Panorámico · `15740042_3840_2160_60fps.mp4` | 1.500 | 43 | 41 | 21 | 0 | 20 |
| Carretera vertical · `12893008_1080_2320_29fps.mp4` | 285 | 5 | 5 | 5 | 0 | 0 |

Los conteos son trayectorias estimadas. En el panorámico se recuperaron 9/9 cascos del control original y 8/10 de los controles adicionales. En ocho regiones negativas hubo dos falsos ocupantes y cero cascos falsos al final. El análisis inicial del panorámico tomó unos 10 minutos en el equipo de referencia; en la prueba de reproducción, el archivo guardado duró 25,3 segundos. [Revisión de votos](reports/training/revision-votos/comparacion.json) · [Seguimiento en carretera](reports/training/road-bike-tracking/comparacion.json) · [Medición de reproducción](reports/training/tracking-occupants-v9/reproduccion-guardada.json)

## Qué modelo se conserva

El detector principal de video es **joint-occupant-negatives-11**, un Faster R-CNN con CNN propia. El último ajuste usó 83 imágenes y 34 regiones negativas revisadas. Se seleccionó el paso 60 por el F1 de ocupante. Las mejoras posteriores de asociación y voto funcionan durante la inferencia.

La configuración publicada necesita **seis archivos** `runs/<nombre>/best.pt`. Conserva sus nombres y rutas.

| Función | Pesos |
|---|---|
| Detector de video y origen que valida el cargador | `joint-occupant-negatives-11`, `joint-reviewed-08-10` |
| Detector de foto y cámara y su origen verificado | `joint-multiscale-02`, `joint-clean640-01` |
| Localización de cabezas y verificación del recorte | `rider-crop-heads-01`, `crop-specialist-01` |

## Estructura del repositorio

| Ruta | Qué encontrarás |
|---|---|
| `training/photo_app.py` | Punto de entrada de la interfaz. |
| `training/predict.py`, `training/models.py` | Carga y comprobación de pesos, inferencia y arquitectura. |
| `training/traffic_video.py`, `training/video_recording.py` | Seguimiento, votación por ocupante, consultas y reproducción guardada. |
| `training/train_occupant_regions.py` | Código del último ajuste del detector principal. Entrenar requiere los datasets preparados. |
| `runs/` | Pesos publicados y registros de entrenamiento. |
| `data/`, `scripts/` | Anotaciones, manifiestos y utilidades para preparar y auditar los datos. |
| `reports/` | Evaluaciones del proyecto e informes locales de la app. |
| `docs/`, `backups/` | Documentación y material histórico de los experimentos. |

Para ejecutar las pruebas desde la raíz:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s training -p "test*.py"
```
