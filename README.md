# moto-detector

Proyecto académico para detectar motocicletas, asociar sus ocupantes, clasificar el uso visible del casco y consultar los conteos de fotos y videos. Los pesos proceden de entrenamientos propios. La aplicación distingue **con casco**, **sin casco** e **indeterminado**. El ajuste de la correa queda fuera de las cuatro clases anotadas.

**Estado al 25 de noviembre de 2026.** El detector para archivos de video `joint-occupant-negatives-11` obtuvo **80,43 % de mAP50 en 5 imágenes de validación**. El seguimiento **v9** usa los mismos pesos propios. Al procesar los 1.500 fotogramas del video panorámico pasó de **7 a 8 de 10 cascos** en los fotogramas de control y conservó **9 de 9** en la revisión original. En las ocho regiones negativas volvieron a aparecer dos falsos ocupantes en el conteo final. [Resultados y ejemplos de v9](reports/training/tracking-occupants-v9/revision.md)

## Arranque rápido en el equipo del proyecto

Abre PowerShell y ejecuta estos comandos. La carpeta del escritorio ya contiene el entorno y los pesos.

```powershell
Set-Location -LiteralPath "$env:USERPROFILE\Desktop\INTELIGENCI AARTIFICIAL - MOTOS"
.\.venv\Scripts\python.exe training\photo_app.py
```

1. Pulsa **Cargar video**. Para repetir la prueba panorámica, deja desmarcada **Video de frente con lluvia (experimental)** y selecciona `15740042_3840_2160_60fps.mp4`. El título de la ventana debe mostrar **joint-occupant-negatives-11 · seguimiento v9**. Este modo se activa en archivos de al menos 1920 píxeles de ancho con relación ancho/alto ≥ 1,45.
2. Deja terminar el video para comparar sus totales con el informe completo. **Detener y guardar** genera un informe parcial. El JSON queda en `reports/videos/<id>/video.json` e incluye modelo, versión, fotogramas, velocidad, conteos y votos por trayectoria.
3. **Cargar foto** y **Usar cámara** utilizan `joint-multiscale-02` en modo general. Las fotos se guardan en `reports/fotos/<id>/` con su imagen anotada y JSON. La casilla de lluvia selecciona `joint-rain-rider-09` para video o cámara.
4. En **Consultar estadísticas** puedes preguntar «¿Cuántos ocupantes sin casco hubo hoy en videos?» o «Resumen en fotos». Cada ejecución guardada cuenta como un análisis, incluso si cargas el mismo archivo dos veces.

## Instalación en otro equipo Windows

El entorno comprobado usa **Python 3.14.5 de 64 bits**, Tkinter, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Pillow 13.0.0 y OpenCV headless 5.1.0.95. La GPU utilizada fue una NVIDIA GeForce RTX 3060. Desde PowerShell, con Git y Python 3.14 instalados, ejecuta:

```powershell
git clone --branch codex/frozen-inference-review-20261124 https://github.com/MAKEBUZ/moto-detector.git
Set-Location moto-detector
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r training\requirements-cu128.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install opencv-python-headless==5.1.0.95
.\.venv\Scripts\python.exe -c "import tkinter, torch, torchvision, cv2; print(torch.__version__, torchvision.__version__, cv2.__version__, 'CUDA:', torch.cuda.is_available())"
```

**El clon de esta rama incluye los seis archivos de pesos para foto, cámara general y video panorámico v9.** Para el modo experimental de lluvia, copia desde el equipo del proyecto los archivos indicados en la última fila. Los conjuntos de entrenamiento y las anotaciones locales de evaluación se restauran por separado. El cargador verifica la huella del modelo de origen. Los nombres de los archivos y de sus carpetas deben coincidir. [Dependencias fijadas](training/requirements-cu128.lock.txt) · [Carga y comprobación de pesos](training/predict.py)

| Uso | Archivos que deben estar bajo la raíz del proyecto |
|---|---|
| Foto y cámara generales | `runs/joint-multiscale-02/best.pt` y su origen `runs/joint-clean640-01/best.pt` |
| Archivo de video general | `runs/joint-occupant-negatives-11/best.pt` y su origen `runs/joint-reviewed-08-10/best.pt` |
| Verificación adicional del video panorámico | `runs/rider-crop-heads-01/best.pt`, su origen `runs/joint-multiscale-02/best.pt` y el clasificador `runs/crop-specialist-01/best.pt` |
| Modo de lluvia opcional | `runs/joint-rain-rider-09/best.pt`, su origen `runs/joint-multiscale-02/best.pt` y la carpeta `data/prepared/rain-rider-02/` con su manifiesto e imágenes |

Una vez copiados, puedes comprobar la carga del detector de video y abrir la ventana. Los comandos se ejecutan desde la raíz del clon.

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'training'); from predict import load_model; m = load_model('joint-occupant-negatives-11'); print(m[3], m[2], m[5])"
.\.venv\Scripts\python.exe training\photo_app.py
```

## Último entrenamiento del detector

**Objetivo.** Reducir las piezas de moto clasificadas como ocupantes. El modelo base `joint-multiscale-02` pasó por un ajuste con 10 fotos revisadas adicionales (`joint-reviewed-08-10`, 6 épocas, tasa 0,000008, mejor época 3). El último ajuste partió de esos pesos propios y actualizó exclusivamente los pesos y el sesgo de la salida «ocupante». El extractor de características, el RPN y la regresión de cajas permanecieron congelados. [Código del ajuste](training/train_occupant_regions.py) · [Arquitectura](training/models.py)

| Dato | Valor del último ajuste |
|---|---|
| Entrenamiento | 83 fotos, 73 de `helmet-separate-v2-expanded-09` y 10 de `reviewed-08-addon`. En total, 428 cajas: 118 motos, 155 ocupantes, 67 cabezas con casco y 88 sin casco. |
| Negativos revisados | 34 regiones de 25 fotos de entrenamiento, con ruedas, faros, carenados y otras piezas. El conjunto incluye 7 propuestas de ocupante sobre recortes. |
| Entradas al ajuste | Regiones originales y reflejadas horizontalmente. En total, 310 ejemplos de ocupante, 546 de las otras clases y 68 de piezas de moto. |
| Detector | Faster R-CNN con CNN propia de 128 canales de salida, paso espacial 8, tamaño configurado de 640 píxeles y presupuesto de propuestas ×4. RoIAlign obtiene características de 7 × 7 por región. |
| Optimización | Adam, tasa 0,00035, 100 actualizaciones sobre el banco completo de características. La pérdida penaliza aceptar piezas como ocupantes y descartar personas reales. Dos términos regularizan el cambio de puntuación en las otras clases (0,1) y en personas (0,05). |
| Selección | Validación cada 10 pasos. Se priorizó F1 de ocupante, después mAP50 y finalmente el primer paso en caso de empate. **Paso elegido 60**, F1 **82,35 %**, pérdida **0,11674**. El mayor mAP50 observado fue 81,44 % en los pasos 10 a 40. |
| Pesos finales | `runs/joint-occupant-negatives-11/best.pt`. SHA-256 `0fcf5d2ed15c220eed2732f86c93cb7f2a63e624f7e792bdc7a03a0e291909ed`. |

El historial de los diez controles está en `runs/joint-occupant-negatives-11/history.json`. En la misma carpeta están `protocol.json`, con la procedencia de las regiones, y `photo-comparison.json`, con la comparación de las fotos reservadas. El ajuste usa CUDA y crea su carpeta de salida solo si está libre. Para reproducirlo en una **copia de trabajo separada**, restaura el modelo padre y su origen, los dos datasets preparados con sus imágenes y manifiestos, y ejecuta el comando siguiente. [Regiones utilizadas](reports/training/occupant-negatives-01/selected.json)

```powershell
.\.venv\Scripts\python.exe training\train_occupant_regions.py
```

### Evolución durante el ajuste

Cada paso de Adam utiliza el banco de regiones completo. La pérdida corresponde al objetivo de entrenamiento y las dos métricas se calculan en las cinco imágenes de validación. El paso 60 fue el que mejor equilibró precisión y recobrado para ocupantes.

| Paso | Pérdida | F1 de ocupante | mAP50 |
|---:|---:|---:|---:|
| 10 | 0,16540 | 75,00 % | 81,44 % |
| 20 | 0,15175 | 75,00 % | 81,44 % |
| 30 | 0,14054 | 75,00 % | 81,44 % |
| 40 | 0,13120 | 75,00 % | 81,44 % |
| 50 | 0,12336 | 75,00 % | 80,01 % |
| **60** | **0,11674** | **82,35 %** | **80,43 %** |
| 70 | 0,11112 | 77,78 % | 80,43 % |
| 80 | 0,10630 | 77,78 % | 80,43 % |
| 90 | 0,10214 | 77,78 % | 80,43 % |
| 100 | 0,09852 | 73,68 % | 80,43 % |

### Validación del checkpoint seleccionado

Cinco imágenes contienen 25 cajas. AP50 integra la curva precisión–recobrado en 101 puntos con IoU ≥ 0,50. El mAP50 es la media de las cuatro AP. Para los aciertos, falsos positivos y omisiones de la tabla se usó además confianza ≥ 0,50. [Cálculo de AP](training/metrics.py) · [Conteos por clase](training/evaluate_reserved.py)

| Clase | Cajas reales | AP50 | Aciertos | Falsos positivos | Omisiones | Precisión | Recobrado |
|---|---:|---:|---:|---:|---:|---:|---:|
| Motocicleta | 7 | 80,53 % | 5 | 2 | 2 | 71,43 % | 71,43 % |
| Persona ocupante | 9 | 78,01 % | 7 | 1 | 2 | 87,50 % | 77,78 % |
| Cabeza con casco | 3 | 100,00 % | 3 | 1 | 0 | 75,00 % | 100,00 % |
| Cabeza sin casco | 6 | 63,17 % | 3 | 1 | 3 | 75,00 % | 50,00 % |

**mAP50 global: 80,43 %.** En las **14 fotos reservadas**, con el flujo de asociación, el resultado fue:

| Clase | Aciertos | Falsos positivos | Omisiones |
|---|---:|---:|---:|
| Motocicleta | 16 | 3 | 1 |
| Persona ocupante | 17 | 5 | 3 |
| Cabeza con casco | 5 | 0 | 3 |
| Cabeza sin casco | 8 | 1 | 4 |

El detector enlazó correctamente **11 de 20** relaciones completas moto–persona–cabeza. Frente al modelo padre conservó los 17 ocupantes correctos y redujo de 7 a 5 los falsos ocupantes. En esas mismas fotos, el modelo general de fotografía obtuvo 18 ocupantes correctos, 3 falsos ocupantes y 12 relaciones completas.

### Prueba de video con los mismos pesos

El seguimiento v5 busca en el fotograma actual un casco verificado recientemente en la misma trayectoria y comprueba el recorte con el clasificador local. La app puede mostrar «casco visible» mientras espera evidencia para clasificar la trayectoria. El cambio se aplicó durante la inferencia con los pesos del paso 60.

El seguimiento v6 añade flujo óptico con comprobación de ida y vuelta, clasificación del recorte actual y una nueva localización con el detector de cabezas. Una reasociación necesita una caja de persona detectada en el fotograma y una trayectoria previa confirmada. Para las cajas recuperadas se limita el área de la cabeza al 20 % del área del cuerpo, valor redondeado del percentil 95 de 139 parejas confirmadas de entrenamiento.

El seguimiento v7 conserva el vínculo al salir por un borde. Parte de un casco verificado y sigue por separado puntos del casco y del torso. Comprueba que ambas zonas siguen visibles, conservan su aspecto y se mueven juntas. Dibuja la parte visible sobre una trayectoria que ya contaba como «con casco». Estas observaciones se registran por separado de los votos del detector.

| Video panorámico de 1.500 fotogramas | v4 | v5 | v6 | v7 | v8 | v9 |
|---|---:|---:|---:|---:|---:|---:|
| Cascos localizados en los 9 fotogramas marcados | 3/9 | 5/9 | 7/9 | 9/9 | 9/9 | 9/9 |
| Cascos de esas marcas confirmados al terminar | 2/9 | 4/9 | 7/9 | 9/9 | 9/9 | 9/9 |
| Falsos ocupantes confirmados en las 8 regiones negativas revisadas | 2/8 | 2/8 | 2/8 | 2/8 | 2/8 | 2/8 |
| Trayectorias de moto / de ocupante | 43 / 40 | 43 / 40 | 43 / 40 | 43 / 40 | 43 / 40 | 43 / 41 |
| Trayectorias con casco / sin casco / indeterminadas | 14 / 0 / 26 | 16 / 0 / 24 | 18 / 0 / 22 | 18 / 0 / 22 | 19 / 0 / 21 | 20 / 0 / 21 |
| Cascos confirmados en los cinco fotogramas de control | 5/10 | 5/10 | 5/10 | 5/10 | 7/10 | 8/10 |

En v6 se recuperaron las marcas de los fotogramas 352, 400 y 931. En v7, el casco del fotograma 680 vuelve a la trayectoria 27 (IoU 0,678) y el del 837 vuelve a la 31 (IoU 0,564). El mínimo exigido es 0,50. Los dos ocupantes ya contaban como «con casco» en v6. En las ocho regiones negativas, tres presentaron una asociación en el fotograma marcado y dos llegaron al conteo final en ambas versiones. Esas dos trayectorias, en los fotogramas 1001 y 1297, quedaron indeterminadas. [Comparación por marca](reports/training/tracking-occupants-v7/comparison.json) · [Vista de los dos bordes](reports/training/tracking-occupants-v7/vista-bordes.jpg)

En otros cinco fotogramas del mismo video, las versiones v4 a v7 confirmaron 5 de 10 cascos visibles. Las nueve marcas se usaron para diagnosticar los fallos, por lo que el 9/9 mide la corrección en este video. Una trayectoria puede fragmentarse y asignar dos identificadores a la misma persona. En este equipo se midieron 4,12 fotogramas/s con v6 y 2,74 con v7. [Comprobación adicional](reports/training/tracking-occupants-v7/additional-frames.json) · [Auditoría visual de las 135 asociaciones añadidas](reports/training/tracking-occupants-v7/revision-visual.md)

En v8, dos detectores propios buscan el casco en recortes de diferente tamaño alrededor del ocupante. Durante una pérdida breve del cuerpo se exige continuidad visual del casco y del torso y se verifica la cabeza en el fotograma actual. Se recuperaron el casco del fotograma 750 (trayectoria 25) y el de la izquierda del 1350 (trayectoria 51). La trayectoria 25 ya contaba como «con casco». La 51 pasó de «indeterminado» a «con casco». Los casos pendientes son el 150, el conductor del 450 y el motorista central del 1350. Estas marcas se usaron para desarrollar el ajuste y miden el resultado sobre el mismo video. [Comparación de los cinco casos](reports/training/tracking-occupants-v8/cinco-controles.jpg) · [Resultados y revisión visual](reports/training/tracking-occupants-v8/revision.md)

En v9 se recuperó al ocupante de casco blanco parcialmente visible en el fotograma 150. Sus 140 observaciones corresponden a una sola trayectoria. Quedan pendientes el conductor del 450 y el motorista central del 1350. [Los tres casos](reports/training/tracking-occupants-v9/tres-cascos.jpg) · [Comparación por fotograma](reports/training/tracking-occupants-v9/additional-frames.json)

Para ejecutar las pruebas del proyecto y repetir las comparaciones en la máquina de trabajo, utiliza los siguientes comandos.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s training -p "test*.py"
.\.venv\Scripts\python.exe training\evaluate_rider_tracking.py --temporal reports\videos\2da824dd4ddd4993bd19bc091c7f2c18\video.json
.\.venv\Scripts\python.exe training\evaluate_rider_tracking.py --motion reports\training\tracking-occupants-v5\video.json
.\.venv\Scripts\python.exe training\evaluate_rider_tracking.py --edge reports\training\tracking-occupants-v6\video.json
.\.venv\Scripts\python.exe training\evaluate_rider_tracking.py --consensus reports\training\tracking-occupants-v7\video.json
.\.venv\Scripts\python.exe training\evaluate_rider_tracking.py --partial reports\training\tracking-occupants-v8\video.json
```

Las comparaciones requieren el video original en la ruta guardada en el informe de control, el archivo local `runs/joint-occupant-negatives-11/paired-review.json` y los pesos del modo panorámico.

## Historial de experimentos anteriores

Las conclusiones que siguen describen el estado del proyecto en la fecha de cada experimento.

<details>
<summary>Ver datasets, entrenamientos y auditorías anteriores</summary>

## Resultados acumulados

### Modelos iniciales

| Modelo | Datos de entrenamiento / validación | Resultado de validación |
|---|---|---|
| Clasificador CNN de recortes de cabeza | 2889 / 628 recortes | Macro-F1: 91,00 %; exactitud: 91,72 % |
| Detector inicial de tres clases | 1583 / 347 imágenes | mAP@0,50: 69,12 % |

El detector inicial reconoce **moto con ocupante** como una sola clase, además de cabeza con casco y sin casco. Sus resultados **no son directamente comparables** con los del detector nuevo: cambian las clases, las anotaciones y la validación. El clasificador requiere un recorte de cabeza; no detecta personas por sí solo. [Informe inicial](reports/training/README.md).

### Detector de cuatro clases

Clases: **motocicleta, persona ocupante, cabeza con casco y cabeza sin casco**. El dataset de esta etapa tiene 44 imágenes de entrenamiento, 246 cajas y 89 cabezas; la validación contiene 5 imágenes, 25 cajas y 9 cabezas (3 con casco y 6 sin casco). Se conservaron las particiones originales y se excluyeron imágenes con cobertura incompleta o incertidumbre.

Las cifras siguientes corresponden al checkpoint seleccionado por **mAP global de validación**, no al máximo independiente de cada clase.

| Experimento | Train | Épocas | Mejor época | Moto AP | Persona AP | Con casco AP | Sin casco AP | mAP@0,50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Piloto de cuatro clases | 14 | 20 | 18 | 42,57 % | 35,30 % | 0 % | 0 % | 19,47 % |
| Dataset ampliado | 25 | 20 | 4 | 42,57 % | 44,55 % | 0 % | 0 % | 21,78 % |
| Mayor exposición, paso 16 | 25 | 150 | 72 | 68,98 % | 59,98 % | 0 % | 0 % | 32,24 % |
| **Paso 16 + variaciones fotométricas** | **25** | **150** | **105** | **69,70 %** | **55,60 %** | **33,66 %** | **16,83 %** | **43,95 %** |
| Paso 8 + variaciones fotométricas | 25 | 150 | 96 | 57,43 % | 45,64 % | 33,66 % | 16,83 % | 38,39 % |
| Paso 8 + propuestas ×4 | 25 | 150 | 136 | 57,43 % | 58,25 % | 33,66 % | 50,50 % | 49,96 % |
| Lote 04 + paso 8 + propuestas ×4 | 35 | 150 | 108 | 63,54 % | 56,35 % | 100 % | 37,87 % | 64,44 % |
| Lote 05 + paso 8 + propuestas ×4 | 44 | 150 | 75 | 62,38 % | 49,22 % | 33,66 % | 43,76 % | 47,26 % |

Todos estos experimentos usan la misma validación de cinco imágenes. Los de 150 épocas mantienen tasa de aprendizaje constante de 0,0003; los de 20 épocas usaban planificación coseno. La política fotométrica aplica brillo y contraste entre 0,85 y 1,15 y desenfoque suave con probabilidad del 25 %, exclusivamente en entrenamiento. El experimento de paso 8 mantiene esa política y los demás parámetros; duplica la resolución espacial del mapa de características por eje.

**Conclusión de esta etapa:** con 44 imágenes mejora el AP sin casco de 37,87 % a 43,76 %, pero el mAP global cae de 64,44 % a 47,26 %. A confianza ≥0,5 sigue habiendo cero aciertos sin casco; desaparece el único falso positivo por encima de ese umbral. A confianza 0,05 se mantienen 3/6 aciertos, con un mejor orden de puntuaciones y 8 falsos positivos frente a 7. No basta para habilitar conteo. Se mantienen 150 épocas, pero las actualizaciones suben de 2700 a 3300; no se aisló el efecto de los datos del número de pasos.

El checkpoint anterior de propuestas ×4 obtuvo mAP@0,50 de 26,55 % en ocho imágenes test, con recall sin casco de 0 % a confianza 0,5. Ese resultado pertenece al modelo anterior; el nuevo no se evaluó en test. [Evaluación independiente](reports/independent-evaluation/results-01/README.md).

### Control a igual número de actualizaciones y dos semillas

Se completaron cuatro entrenamientos nuevos desde cero, cada uno con **1950 actualizaciones exactas**, conservando configuración y validación. Se evaluó `last.pt` del paso fijo, sin elegir la mejor época.

| Semilla | AP sin casco: 35 imágenes | AP sin casco: 44 imágenes | TP/FP/FN a confianza 0,5: 35 → 44 |
|---|---:|---:|---|
| 20261124 | 8,42 % | 16,83 % | 0/0/6 → 1/0/5 |
| 20261125 | 0 % | 0 % | 0/0/6 → 0/1/6 |

**La mejora no se repitió con la segunda semilla; no se considera estable.** No comparar estos resultados de paso fijo con los mejores checkpoints de la tabla anterior. Se igualaron actualizaciones, pero el loader de 35 imágenes tiene un minibatch unitario por época: procesó 3792 imágenes frente a 3900 con el dataset de 44. Dos semillas y cinco imágenes de validación reutilizadas no permiten confirmar generalización. El conteo permanece deshabilitado. [Informe y protocolo](reports/training/fixed-updates-01/README.md).

### Auditoría de propuestas y muestreo (sin entrenamiento nuevo)

En los cuatro checkpoints de paso fijo hay propuestas decodificadas que cubren las seis cabezas sin casco, pero solo 2/6, 1/6, 0/6 y 1/6 sobreviven al top 2000 por objectness; NMS no reduce más esa cobertura. En la reproducción controlada de train, sin casco recibe solo 3,9–4,3 % de los positivos RPN, mientras cada cabeza obtiene positivos RoI en todos los sorteos. Algunas propuestas válidas de validación se clasifican como fondo. Esto prioriza estudiar el muestreo RPN por objeto y registrar sus conteos reales antes de otro cambio. Es diagnóstico con pesos finales, no reconstrucción del muestreo histórico. [Informe detallado](reports/training/fixed-updates-01/positive-sampling-v2/README.md).

### Experimento de reserva de positivos RPN por objeto

Se implementó `--rpn-object-reserve`, desactivado por defecto: conserva el cupo de positivos/negativos y reemplaza positivos redundantes para cubrir objetos elegibles omitidos. Se registran las selecciones reales por paso, imagen y objeto. Se ejecutaron cuatro entrenamientos nuevos desde cero, de 1950 actualizaciones exactas, sin pesos preentrenados.

| Train | Semilla | AP sin casco control | Con reserva | TP/FP/FN con reserva a 0,5 |
|---|---|---:|---:|---|
| 35 | 20261124 | 8,42 % | 16,83 % | 0/0/6 |
| 44 | 20261124 | 16,83 % | 0 % | 0/1/6 |
| 35 | 20261125 | 0 % | 0 % | 0/0/6 |
| 44 | 20261125 | 0 % | 8,42 % | 1/2/5 |

La reserva corrigió **1077 omisiones en 18706 presentaciones de cabezas sin casco**, pero no mejoró consistentemente la detección; el mAP global bajó en las cuatro parejas. No se adopta como nueva referencia ni se habilita conteo. Se verificaron 34 pruebas, integración, presupuestos, contadores AdamW, inicializaciones y reproducción de métricas. [Informe y registros](reports/training/object-reserve-01/README.md).

### Comparación de propuestas tras reservar positivos

La cobertura RPN de las seis cabezas sin casco cambia de 2→2, 1→1, 0→1 y 1→1 entre controles y reserva. El ranking sigue descartando la mayoría de las cabezas; algunas propuestas válidas se clasifican como fondo. En un caso, una propuesta con puntuación sin casco 0,885 pasa de IoU 0,577 a 0,438 tras ajustar la caja y pierde el acierto. La reserva solo incrementó el volumen total de positivos sin casco un 1,9–2,1 % dentro de esas ejecuciones. No se entrenó ni modificó ningún peso para esta revisión. [Informe con ejemplo visual](reports/training/object-reserve-01/proposal-comparison-v2/README.md).

### Prueba congelada: cajas de cabeza sin regresión

Se compararon los ocho checkpoints usando las mismas puntuaciones y filtros, reemplazando únicamente las cajas ajustadas de las dos clases de cabeza por sus propuestas RPN. El AP sin casco mejoró en 2/8 modelos, empeoró en 1/8 y empató en 5/8. Se recuperó el caso con puntuación 0,885 cuya regresión llevaba IoU de 0,577 a 0,438, pero otro control perdió un acierto y otro añadió dos falsos positivos a confianza 0,5. **No se desactiva la regresión de forma general.** No hubo entrenamiento ni cambios de pesos o despliegue. [Resultados completos](reports/training/object-reserve-01/head-box-ablation/README.md).

### IoU de cabezas por tamaño y clase

Se comparó la misma propuesta antes/después del ajuste, con pesos y puntuaciones fijos, sobre train y val de los ocho checkpoints. En entrenamiento el IoU medio mejora en ambas clases y todos los tamaños. En validación, sin casco grande baja de 0,587 a 0,547, pero solo hay cinco pares cabeza–modelo con propuesta cercana entre 32 posibles; el grupo pequeño tiene un único par y puntuación muy baja. No hay evidencia suficiente para desactivar la regresión según tamaño. Se verificaron 724 pares, estados intactos y concordancia con los diagnósticos previos; no se entrenó. [Informe por tamaños y clases](reports/training/object-reserve-01/regression-by-size/README.md).

### Ranking RPN de cabezas sin casco

En los ocho checkpoints, las seis cabezas sin casco de validación tienen anclas asignables como positivas y cajas decodificadas que las cubren. El top 2000 conserva solo 0–2 cabezas. Las puntuaciones de anclas positivas son mucho mayores en train que en val; 60–76 % del top 2000 tiene IoU <0,5 con cualquier anotación (esto no certifica fondo). El cuello de botella observado es la priorización, con variación entre semillas, no una ausencia general de anclas. Se proponen pruebas congeladas de presupuesto antes de otro entrenamiento; no se cambió el detector. [Informe RPN](reports/training/object-reserve-01/rpn-ranking/README.md).

### RPN congelado: 2000 frente a 4000 propuestas

Se compararon los ocho checkpoints cambiando solo el top-k previo a NMS; el límite posterior permaneció en 800. La cobertura de cabezas sin casco aumenta en todos, pero AP mejora en cuatro modelos, baja en uno y empata en tres. A confianza 0,5 solo el control/44/semilla 20261124 gana un acierto (1→2 TP, sin FP); reserva/44/semilla 20261125 añade un FP. Los FP a confianza baja aumentan en los ocho. Se midió tiempo con calentamiento y 20 repeticiones pareadas; hay variabilidad y los datos no equivalen a FPS de video. No se adopta 4000 como configuración global. [Informe de detección y tiempo](reports/training/object-reserve-01/topk-2000-vs-4000/README.md).

### Diagnóstico y asociación

- En el piloto humano de siete imágenes, el sistema propuso correctamente **5 de 11 vínculos confirmados cabeza→moto**. Los demás fallaron por localización o abstención ante clases contradictorias. La comparación usa el contexto antiguo moto+ocupante; no evalúa todavía una cadena completa cabeza→persona→moto.
- En una prueba de memorización con **4 imágenes y 7 cabezas**, el detector alcanzó AP@0,50 del **100 % en ambas clases** sobre la propia muestra entrenada, desde el paso 275. Esto confirma capacidad de aprender esos ejemplos, **no generalización**.
- Tras 150 épocas sin variaciones fotométricas, detectaba 39/45 cabezas de entrenamiento con clase correcta y confianza ≥0,5, frente a 0/9 en validación. Se observó una brecha compatible con sobreajuste.
- La inspección visual mostró fallos tanto en cabezas pequeñas como grandes. En algunos casos la propuesta localiza la cabeza, pero el clasificador la rechaza como fondo. El tamaño no explica por sí solo el problema.

### Límites y decisión

AP@0,50 mide detección a IoU 0,50; **no es exactitud ni COCO mAP@0,50:0,95**. Usa las predicciones del detector desde su umbral interno 0,05, por lo que no equivale al rendimiento con confianza mínima de 0,5. La validación es pequeña y se ha consultado repetidamente para elegir checkpoints; hace falta una evaluación independiente para confirmar cualquier mejora.

Se verificaron inicialización aleatoria, hashes de datos y checkpoints, métricas de cuatro clases y aislamiento de aumentos a entrenamiento. La última batería completó **34 pruebas**. Los pesos propios están en `runs/`, excluido de Git junto con los datasets descargados y preparados; los informes y anotaciones revisadas sí están versionados.

La mejora global todavía no resuelve la detección sin casco. El siguiente análisis es revisar sus fallos en la validación original y ampliar diversidad de entrenamiento; una confirmación independiente requerirá una nueva reserva. No se ha implementado la asociación de cuatro clases ni habilitado conteo, video, asistente de estadísticas o evaluación del casco correctamente abrochado.

## Dataset histórico del lote 05

El lote 05 aportó 9 imágenes completas (3 excluidas por cobertura no declarada). La nueva versión `helmet-separate-v2-expanded-03` contiene **44 imágenes train, 246 cajas y 89 cabezas**; mantiene las mismas 5 imágenes de validación. Se completó un entrenamiento desde cero de 150 épocas sobre esta versión; mejor checkpoint en la época 75. Los modelos anteriores permanecen disponibles. [Detalle y reproducción](docs/dataset-separado-v2-lote-05.md).

## Informes de los experimentos

- [Piloto de cuatro clases](reports/training/separate-v2-pilot-01/README.md)
- [Entrenamiento con dataset ampliado](reports/training/separate-v2-expanded-01/README.md)
- [Prueba de memorización de cabezas](reports/training/head-overfit-01/README.md)
- [Entrenamiento prolongado](reports/training/separate-v2-long-01/README.md)
- [Inspección de cabezas de validación](reports/training/separate-v2-long-01/head-comparison/README.md)
- [Brillo, contraste y desenfoque](reports/training/separate-v2-photo-01/README.md)
- [Características con paso 8](reports/training/separate-v2-stride8-01/README.md)
- [Propuestas ×4](reports/training/separate-v2-proposals4-01/README.md)
- [Ampliación con diez imágenes del lote 04](reports/training/separate-v2-expanded-02/README.md)
- [Entrenamiento con 44 imágenes del lote 05](reports/training/separate-v2-expanded-03/README.md)
- [Revisión de los fallos sin casco](reports/training/separate-v2-expanded-02/nohelmet-review/README.md)
- [Comparación de vínculos revisados](reports/relations-pilot/README.md)
- [Dataset ampliado de cuatro clases](docs/dataset-separado-v2-ampliado.md)

## Documentación de datos y preparación

- [Informe de auditoría](reports/auditoria-dataset.md)
- [Criterios de anotación y preparación](docs/criterios-dataset.md)
- [Resultados automáticos](reports/dataset-audit.json)
- [Recuperación y validación de originales](reports/recuperacion-originales.md)
- [Instrucciones para repetir las descargas](docs/recuperar-originales.md)
- [Fuentes y versiones fijadas](data/sources.lock.json)
- [Resultado de la depuración y particiones](reports/preparation/README.md)
- [Guía del dataset preparado](docs/dataset-preparado.md)
- [Primer entrenamiento y resultados](reports/training/README.md)
- [Entrenadores e inferencia desde cero](training/README.md)
- [Revisión de errores y asociación cabeza–moto](reports/error-review/README.md)

## Reproducir la auditoría

Requiere Git y Node.js, sin paquetes adicionales. Desde la raíz del proyecto:

```powershell
git clone --depth 1 https://github.com/Valery-Rosero/DataSetsIA.git data/sources/DataSetsIA
node scripts/audit-dataset.mjs
powershell -ExecutionPolicy Bypass -File scripts/preview-crops.ps1
```

El JSON registra el commit auditado. Para reproducir exactamente un informe, utilizar ese commit de la fuente. La muestra visual se genera en Windows con System.Drawing; no sustituye la revisión de todos los datos.

`data/sources/` está excluido de Git. Los scripts de auditoría no modifican las etiquetas originales. Las copias en `reports/upstream-reference/` se conservan como evidencia del código y los conteos publicados; no se deben ejecutar desde allí.

El repositorio fuente contiene etiquetas y recortes; las imágenes originales ya se recuperaron por separado en `data/sources/originals/`. No ejecutar su unificador sobre la única copia disponible: elimina su salida anterior antes de regenerarla.

### Clasificación y ajuste de las propuestas adicionales con 4000

Diagnóstico congelado de los ocho modelos: las propuestas adicionales aportan 6 TP y 27 FP sin casco a confianza 0.05 (observaciones repetidas entre modelos). En 25 de esos FP la clase dominante es fondo; ninguno tenía IoU sin casco ≥0.5 antes de la regresión. El contrafactual sin ajuste de cajas sin casco no reduce los FP en ningún modelo. A confianza 0.5 aparecen un TP y un FP adicionales. Se mantiene 2000 como referencia, sin entrenamiento ni cambios de pesos. [Desglose, protocolo y limitaciones](reports/training/object-reserve-01/additional-proposals/README.md).

### Revisión visual y filtros congelados

El falso positivo sin casco de confianza 87.9% corresponde al asiento de una moto. Se revisaron los 27 FP adicionales y se compararon filtros de clase ganadora y posición respecto a personas, con RPN2000/4000 y los ocho checkpoints. Reducen FP pero pierden demasiados TP; no se activan por defecto ni se afirma que el detector esté listo para uso real. No hubo entrenamiento. [Informe y galerías](reports/training/object-reserve-01/visual-review/README.md). Respaldo local de los mejores resultados: rama `codex/backup-best-results-20261124`, commit `bcf424d`, checkpoints y datasets incluidos con SHA256.

### Negativos difíciles pendientes de revisión

Preparadas seis imágenes de seis grupos nuevos de entrenamiento: asientos, ropa y espejos/piezas. Un único JSON conserva la revisión de regiones y las cuatro clases habituales. Sin entrenamiento ni cambios de validación. [Instrucciones](data/annotations/train-hard-negatives-v2-01/LEEME.md) · [Selección y trazabilidad](reports/separate-v2/hard-negatives-selection/README.md).

### Negativos difíciles revisados: dataset de 50 imágenes

Las seis imágenes del lote fueron válidas y completas. Nuevo dataset `helmet-separate-v2-expanded-04`: 50 imágenes de entrenamiento y 267 cajas, con las mismas cuatro clases. Se conservan seis regiones no-cabeza como trazabilidad (tres de ropa y tres piezas), sin clases extra. Validación idéntica por SHA256: cinco imágenes y 25 cajas. No hubo entrenamiento. [Preparación y comprobaciones](docs/dataset-separado-v2-negativos-01.md).

### Entrenamiento desde cero: 50 frente a 44 imágenes, dos semillas

Dos entrenamientos nuevos sin pesos preentrenados, comparados con controles44 verificados:1950 actualizaciones y3900 presentaciones por ejecución; mismo detector, aumentos y muestreo original. AP50 sin casco: semilla20261124,16.83% →16.83%; semilla20261125,0% →33.66%. A confianza0.05, TP/FP/FN:1/2/5 →1/5/5 y0/6/6 →2/4/4. A0.5 ambos modelos nuevos producen1 TP,0 FP y5 FN. No demuestra mejora consistente ni preparación para uso real; también empeora con casco. Sin cambios en validación, test o modelo por defecto. [Informe, protocolo y métricas](reports/training/hard-negatives-fixed1950-01/README.md).

### Auditoría congelada de negativos difíciles

Comparadas las seis regiones revisadas con controles44/modelos50 y ambas semillas. En replay de20 sorteos por imagen, el RPN no selecciona anclas negativas cercanas (IoU≥0.5) en113/120 y109/120 sorteos de los modelos50. RoI sí muestrea varias donde existen propuestas, pero carece de ellas en una o dos regiones según semilla. Mejora local parcial; no demuestra generalización ni reconstruye los sorteos históricos. Pesos y buffers intactos; no hubo entrenamiento. [Informe y sensibilidad aIoU0.3](reports/training/hard-negatives-fixed1950-01/negative-region-audit/README.md).

### Registro de exposición real durante entrenamiento

Implementado `--log-negative-exposure`: observa las selecciones reales RPN/RoI sin cambiar el muestreador, con regiones transformadas junto a las imágenes. Dos ejecuciones completas desde cero,50 imágenes y1950 pasos, semillas20261124/20261125:936 registros por semilla. Exposición con negativos cercanos (IoU≥0.5): RPN38/468 y26/468; RoI298/468 y250/468. El espejo Image_00655 casi nunca se selecciona. AP50 sin casco final8.42% y0%; no se promueven estos modelos. Se detectó falta de reproducibilidad bit a bit en CUDA incluso al repetir sin registro; las pruebas verifican que el observador conserva selecciones, pérdidas de forward y RNG.43 pruebas pasan. [Informe y registros](reports/training/negative-exposure-02/README.md).

### Reproducibilidad GPU: dos repeticiones con la misma semilla

Dos procesos nuevos con semilla20261124,50 imágenes,1950 actualizaciones y muestreador original confirman falta de reproducibilidad bit a bit en la configuración actual. Entradas, orden, aumentos y etiquetas idénticos durante los1950 pasos; primera diferencia de gradientes/estado en paso1 y pérdidas/RoI en paso2. AP50 sin casco8.42% frente a4.21%; aconfianza0.05,1 TP/1 FP frente a1 TP/5 FP. No se selecciona el resultado favorable. Antes de modificar el muestreo se recomienda un control explícitamente determinista. Sin pesos preentrenados ni cambios de validación. [Informe, protocolo y trazas](reports/training/gpu-reproducibility-01/README.md).

### GPU determinista: dos ejecuciones completas desde cero

Activado el modo determinista estricto, TF32 desactivado y RoIAlign de referencia eager de torchvision: el backward CUDA nativo no admite determinismo estricto en esta instalación Windows. Se mantiene el muestreador original. Dos procesos nuevos con semilla 20261124, 50 imágenes y 1950 actualizaciones cada uno producen pérdidas, entradas, RNG y selecciones idénticos en todos los pasos; gradientes y estado coinciden en los 88 puntos registrados. Pesos finales, estado AdamW, predicciones y registros de exposición también son exactamente iguales. Sin pesos preentrenados ni cambios en validación o test.

Ambas repeticiones: AP50 sin casco **38,83 %**, mAP50 global **37,57 %**. Sin casco, a confianza 0,05: 4 TP, 8 FP y 2 FN; a 0,5: 1 TP, 3 FP y 5 FN. Se compara last.pt a 1950 pasos, sin seleccionar una época favorable. El determinismo queda comprobado para esta máquina y versiones; no demuestra generalización ni preparación para uso real. Esta condición numérica nueva será el control para un futuro cambio de muestreo. 46 pruebas pasan. [Informe y trazas](reports/training/gpu-deterministic-01/README.md).

### Reserva de negativos revisados en modo determinista

Variante desde cero, semilla 20261124 y 1950 pasos: exposición RPN cercana 32/468 → 468/468. Sin casco, AP50 38,83% → 8,42%; a confianza 0,05 los TP/FP/FN pasan de 4/8/2 a 1/5/5. Con casco, AP50 8,42% → 66,34%, TP/FP/FN 1/5/2 → 2/2/1. No se adopta: pierde detecciones sin casco. Muestreador nuevo opcional, sin pesos preentrenados ni cambios de validación. [Informe](reports/training/reviewed-negative-reserve-01/README.md).

### Diagnóstico congelado de la reserva de negativos

Las tres detecciones sin casco perdidas ya carecen de propuestas RPN con IoU ≥0,5: cobertura 4/6 → 1/6. No existen candidatos ajustados suficientes antes de NMS para recuperarlas. Aumenta la puntuación de fondo sobre propuestas distintas, lo que no aísla un deterioro del clasificador. Pesos y buffers intactos, sin entrenamiento ni cambios de validación. [Informe por cabeza](reports/training/reviewed-negative-reserve-01/head-diagnosis/README.md).

### Localización de pérdidas RPN con pesos congelados

Las tres cabezas perdidas por la reserva aún tienen 11, 87 y 107 cajas ajustadas cercanas, pero sus mejores puestos de objectness son 2095, 2435 y 2132: quedan fuera del top-2000 antes de NMS. La supresión no es el punto de pérdida. Propuestas reconstruidas exactamente, pesos y buffers intactos; sin entrenamiento ni cambios de configuración. [Diagnóstico por etapa](reports/training/reviewed-negative-reserve-01/rpn-diagnosis/README.md).

### Presupuesto RPN congelado: control y reserva de negativos

Pre-NMS 2000 → 4000, post-NMS fijo en 800: a confianza 0,05, sin casco, control TP/FP/FN 4/8/2 → 4/14/2; variante 1/5/5 → 3/12/3. AP50 variante 8,42% → 33,66%; control 38,83% sin cambio. A 0,5 la variante pasa de 0/0/6 a 2/1/4. Se recuperan dos cabezas, con más FP. No se activa por defecto; pesos intactos, sin entrenamiento. [Informe](reports/training/reviewed-negative-reserve-01/budget-comparison/README.md).

### Diagnóstico con localización conocida

Con las mismas nueve cajas reales de cabeza, ambos modelos eligen fondo en 7/9. Al comparar solo casco/sin casco, el control acierta 6/9 y la variante 7/9; es una prueba condicionada, no precisión de detección. Sugiere investigar reconocimiento cabeza/fondo y contexto, además de propuestas. Sin entrenamiento ni cambios de pesos. [Diagnóstico y nueva línea de trabajo](reports/training/reviewed-negative-reserve-01/oracle-heads/README.md).

### Prueba anterior en lluvia

- **Video de frente con lluvia (experimental)** activa el detector local entrenado con escenas de lluvia y la comparación visual con seis ejemplos de casco del primer clip. Marca la casilla antes de cargar un video o abrir la cámara. En el segundo clip de prueba se localizaron 6 de 16 apariciones marcadas en ocho fotogramas y se dibujaron otras dos cajas fuera de esas referencias.

### Detector de 70 fotos y especialista de recortes

Respaldo de pesos propios en [backups/detector70-especialista](backups/detector70-especialista). Para reproducir los scripts, copiar detector-last.pt a runs/paired-n70-s20261124/last.pt y especialista-best.pt a runs/crop-specialist-01/best.pt. Se usan los checkpoints del detector al paso 1950 y del especialista en la época 20.

El detector solo obtuvo 53,72 % de mAP50 en validación y 45,84 % en seis escenas reservadas. A confianza 0,5, el especialista con 2000 propuestas recuperó 4/9 cabezas en validación con dos falsos positivos. Con 4000 recuperó 5/9 con cuatro falsos positivos. Se conserva 2000 como referencia y 4000 como variante de mayor cobertura. El cambio de presupuesto usa los mismos pesos. [Comparación](reports/training/paired-70-73/rpn-trace/specialist-4000.json).

</details>

### Ajuste con ocupantes solapados

Se entrenó `joint-crowded-riders-12` a partir de los pesos propios del detector de v9. Durante 60 actualizaciones se combinaron una imagen con varios ocupantes confirmados sobre la misma moto y una imagen aleatoria del conjunto de 73. Se ajustaron la red de propuestas y el predictor de cajas y clases. El paso 30 se seleccionó por F1 de ocupante y después por mAP50 en validación.

En seis fotos reservadas el mAP50 pasó de 80,57 % a 74,94 %. Ambas variantes acertaron cinco de nueve vínculos de cabeza y ocupante. En los 1.500 fotogramas del video, los cascos de control pasaron de 8/10 a 7/10 y los originales confirmados de 9/9 a 4/9. Los falsos ocupantes confirmados siguieron en 2/8. Se conservó el casco blanco del 150, se perdió el del borde en el 750 y quedaron pendientes el conductor del 450 y el centro del 1350. La app sigue usando v9.

Solo dos de las 25 fotos de entrenamiento con varios ocupantes tenían dos cajas de persona sobre la misma moto con IoU de al menos 0,50. [Entrenamiento y validación](reports/training/joint-crowded-riders-12/training.json) · [Repetición del video](reports/training/joint-crowded-riders-12/additional-frames.json)

### Ajuste de v9 con las 16 fotos frontales revisadas

Se partió del detector propio de v9 y se ajustó su predictor de cajas y clases con una foto revisada y una anterior por actualización. El checkpoint del paso 16 obtuvo el mismo F1 de ocupante y mAP50 de validación que los de los pasos 32, 48 y 64. En cinco fotos reservadas, ambos detectores lograron 84,41 % de mAP50 y acertaron 6/8 vínculos. En el video de 1.500 fotogramas, el candidato conservó los mismos 8/10 cascos de control y los 2/8 falsos ocupantes confirmados. En la revisión original localizó 8/9 en el fotograma y confirmó 5/9 al final, frente a 9/9 de v9. El conductor del 450 y el casco central del 1350 siguen pendientes. La app conserva v9. [Comparación por marca](reports/training/joint-v9-frontal-13/comparison.json) · [Los diez controles](reports/training/joint-v9-frontal-13/additional-frames.json)
