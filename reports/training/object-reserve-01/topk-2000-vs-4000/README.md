# RPN congelado: 2000 frente a 4000 propuestas antes de NMS

## Diseño

Se evaluaron los ocho checkpoints del paso 1950, controles y variantes con reserva, con datasets de 35/44 imágenes y ambas semillas. Solo cambió temporalmente el presupuesto RPN de evaluación **antes de NMS**, de 2000 a 4000. El límite después de NMS siguió en 800; los pesos, anclas, puntuadores, regresión, umbrales, NMS y máximo de 100 detecciones finales permanecieron iguales. El presupuesto de entrenamiento tampoco cambió. Al terminar se restauró el valor original y no se modificó el predictor del proyecto.

No se entrenó ni se usó test. La rama 2000 reprodujo el mAP de los checkpoints. Se verificaron hashes de datasets/checkpoints, pesos y buffers intactos y configuración restante idéntica. El protocolo de métricas y tiempos se guardó antes de ejecutar el benchmark.

## Detección sin casco

Hay seis cabezas sin casco en cinco imágenes de validación reutilizadas. AP@0,50 usa el umbral interno 0,05; TP/FP/FN de la tabla usan confianza ≥0,5 e IoU ≥0,5 con matching uno a uno. La cobertura RPN es por referencia, no recall uno a uno.

| Train | Semilla | Modelo | AP 2000 → 4000 | TP/FP/FN 2000 → 4000 a 0,5 | Cobertura RPN 2000 → 4000 |
|---|---|---|---|---|---|
| 35 | 20261124 | control | 8.42 % → 28.05 % | 0/0/6 → 0/0/6 | 2/6 → 4/6 |
| 35 | 20261124 | reserve | 16.83 % → 8.42 % | 0/0/6 → 0/0/6 | 2/6 → 3/6 |
| 44 | 20261124 | control | 16.83 % → 40.88 % | 1/0/5 → 2/0/4 | 1/6 → 5/6 |
| 44 | 20261124 | reserve | 0.00 % → 13.47 % | 0/1/6 → 0/1/6 | 1/6 → 4/6 |
| 35 | 20261125 | control | 0.00 % → 0.00 % | 0/0/6 → 0/0/6 | 0/6 → 3/6 |
| 35 | 20261125 | reserve | 0.00 % → 0.00 % | 0/0/6 → 0/0/6 | 1/6 → 3/6 |
| 44 | 20261125 | control | 0.00 % → 0.00 % | 0/1/6 → 0/1/6 | 1/6 → 2/6 |
| 44 | 20261125 | reserve | 8.42 % → 11.22 % | 1/2/5 → 1/3/5 | 1/6 → 3/6 |

Con 4000 la cobertura de propuestas mejora en los ocho modelos; el AP sin casco mejora en cuatro, empeora en uno y empata en tres. Solo un modelo gana un acierto a confianza 0,5: control/44/semilla 20261124 pasa de 1 a 2 TP sin FP. Reserva/44/semilla 20261125 añade un FP, sin ganar TP a ese umbral. No hay mejora estable en ambas semillas.

## Falsos positivos con confianza baja

| Train / semilla / modelo | TP/FP/FN a 0,05 con 2000 | Con 4000 |
|---|---|---|
| 35 / 20261124 / control | 1/3/5 | 2/8/4 |
| 35 / 20261124 / reserve | 1/0/5 | 1/3/5 |
| 44 / 20261124 / control | 1/2/5 | 3/5/3 |
| 44 / 20261124 / reserve | 0/3/6 | 2/4/4 |
| 35 / 20261125 / control | 0/0/6 | 0/1/6 |
| 35 / 20261125 / reserve | 0/1/6 | 0/4/6 |
| 44 / 20261125 / control | 0/6/6 | 0/12/6 |
| 44 / 20261125 / reserve | 1/3/5 | 2/6/4 |

Los FP a confianza baja aumentan en los ocho modelos. En reserva/35/semilla 20261124 se mantiene un TP pero aparecen tres FP, y AP baja de 16,83 % a 8,42 %. Más propuestas no equivalen a mejores detecciones.

## Tiempo de inferencia

RTX 3060, PyTorch CUDA, modo eval, sin gradientes. Cinco imágenes precargadas en GPU, lotes 2/2/1. Tres pasadas completas de calentamiento por presupuesto, seguidas de 20 repeticiones pareadas alternando el orden 2000/4000 y el orden inicial entre modelos. Se sincroniza CUDA antes/después de cada pasada. No hay hooks de diagnóstico en la medición.

Incluye transformaciones internas y todos los bloques del detector. Excluye lectura de archivos, transferencia CPU→GPU, métricas, capturas de propuestas y escritura de informes. «ms/imagen» es la mediana del tiempo de una pasada completa dividida por cinco: **no es latencia por fotograma aislado ni FPS de video**.

| Train / semilla / modelo | Mediana ms/imagen 2000 | Mediana ms/imagen 4000 | Mediana del cambio porcentual pareado |
|---|---:|---:|---:|
| 35 / 20261124 / control | 9.21 | 10.73 | +20.58 % |
| 35 / 20261124 / reserve | 7.80 | 8.83 | +14.13 % |
| 44 / 20261124 / control | 7.24 | 8.11 | +14.30 % |
| 44 / 20261124 / reserve | 8.19 | 7.90 | +5.92 % |
| 35 / 20261125 / control | 7.20 | 7.94 | +15.30 % |
| 35 / 20261125 / reserve | 9.13 | 10.22 | +11.93 % |
| 44 / 20261125 / control | 9.57 | 10.80 | +9.47 % |
| 44 / 20261125 / reserve | 10.02 | 10.35 | +3.03 % |

La mediana de los cocientes pareados no es el cociente entre las dos medianas. Por eso en reserva/44/semilla 24 la mediana absoluta de 4000 es algo menor, mientras la mediana de las diferencias relativas pareadas es positiva. Los intervalos P10–P90 se solapan ampliamente; no interpretar variaciones pequeñas como ganancias o costes precisos. La carga de GPU, estado térmico y asignación de memoria pueden introducir ruido. Es un microbenchmark local, no una garantía de rendimiento.

| Train / semilla / modelo | P10–P90 de pasada completa 2000 (ms) | P10–P90 4000 (ms) |
|---|---|---|
| 35 / 20261124 / control | 36.09–57.79 | 43.31–66.54 |
| 35 / 20261124 / reserve | 33.36–47.48 | 37.16–48.85 |
| 44 / 20261124 / control | 33.49–43.96 | 37.99–50.70 |
| 44 / 20261124 / reserve | 33.06–49.37 | 35.73–53.49 |
| 35 / 20261125 / control | 32.07–43.75 | 36.95–57.42 |
| 35 / 20261125 / reserve | 42.11–52.35 | 47.01–60.40 |
| 44 / 20261125 / control | 40.63–60.04 | 42.90–61.90 |
| 44 / 20261125 / reserve | 42.09–56.34 | 45.50–60.09 |

## Propuestas reales después de NMS

800 es un límite máximo, no una cantidad fija. Con 2000 previas, las cinco imágenes reciben entre 54 y 616 propuestas finales según checkpoint; con 4000, entre 143 y 800. El aumento amplía el trabajo del clasificador RoI además del filtrado RPN. Los conteos exactos por imagen están en los JSON.

## Decisión

Mantener 2000 como referencia operativa por ahora. No adoptar globalmente 4000 ni seleccionar solo el control favorable: hay mejoras de cobertura, pero no detección consistente entre semillas, y aumentan los falsos positivos de baja confianza. El conteo sigue deshabilitado.

La prueba confirma que el corte del RPN sí impedía pasar cajas útiles, pero también que ampliarlo no resuelve todo: en varios modelos llegan nuevas cabezas y el AP sigue en cero. La siguiente revisión debería centrarse en la clasificación y regresión de esas propuestas nuevas que efectivamente sobreviven al límite de 800, comparándolas con las ya disponibles. Solo después conviene elegir un cambio aislado de entrenamiento; no mezclar presupuesto, umbrales y pérdidas para ajustar estos cinco ejemplos.

## Archivos y verificación

- protocol.json: condiciones fijadas, dispositivo y diseño de tiempos.
- comparison.json: resultados completos de los ocho modelos, cuatro clases, cobertura y tiempos crudos.
- detector-*.json: predicciones y etiquetas por imagen para ambos presupuestos, conteos y métricas.
- Se reprodujeron las métricas de referencia y se verificó restauración de configuración y estado de pesos/buffers.
- 34 pruebas unitarias aprobadas. No hubo entrenamiento ni cambio de despliegue. La validación pequeña y reutilizada no confirma generalización.

Reproducción: `.venv/Scripts/python.exe training/compare_rpn_topk.py`. Rechaza sobrescribir la salida; usar un destino nuevo para repetir. AP50 no es exactitud ni COCO mAP50:95.
