# Cambio de IoU por tamaño y clase de cabeza

## Método fijado

Ocho checkpoints del paso 1950, control/reserva, 35/44 imágenes y dos semillas. Se examinan train y val en modo eval, sin aumentos, con orden de manifiesto y batch 2. No hubo entrenamiento, cambios de pesos, test ni cambios de configuración del detector.

Por cada referencia humana se selecciona la propuesta RPN final de mayor puntuación de su clase real entre las que tienen IoU **antes** del ajuste ≥0,30. Esa misma propuesta, clase y puntuación se usan para comparar IoU antes/después del regresor. No se elige otra caja según su resultado posterior. También se calcula una variante de sensibilidad que exige IoU inicial ≥0,50. Si no hay propuesta elegible, se registra como ausente; no se incluye una caja de fondo arbitraria.

Este emparejamiento usa referencias humanas y es un diagnóstico condicionado de localización, no AP, precisión ni recall. Se realiza antes de los filtros finales y NMS; incluye propuestas de puntuación baja. La comparación end-to-end previa permanece en ../head-box-ablation/.

Tamaño = lado menor de la caja humana en píxeles tras la transformación real del detector: <16, 16–<32, ≥32. No son categorías COCO por área. Mejora/empeora significa ΔIoU >0,01 / <−0,01; cambios menores se consideran dentro de ±0,01 únicamente para resumir. No se modificó el umbral IoU de detección.

## Entrenamiento: diagnóstico sobre imágenes vistas

Los conteos son pares cabeza–modelo; una cabeza puede repetirse en varios checkpoints. Las cabezas añadidas con 44 imágenes aparecen en cuatro modelos y las anteriores en ocho. Son resúmenes descriptivos, no muestras independientes ni inferencia causal.

| Clase | Tamaño | Pares elegibles / total | IoU antes → después | Δ medio | Mejora / empeora / ±0,01 |
|---|---|---|---|---:|---|
| Con casco | <16 px | 79/88 | 0.591 → 0.620 | +0.029 | 53/17/9 |
| Con casco | 16–<32 px | 47/52 | 0.694 → 0.760 | +0.067 | 37/5/5 |
| Con casco | ≥32 px | 117/128 | 0.729 → 0.818 | +0.090 | 93/20/4 |
| Sin casco | <16 px | 79/80 | 0.602 → 0.645 | +0.043 | 54/14/11 |
| Sin casco | 16–<32 px | 42/48 | 0.660 → 0.739 | +0.079 | 31/8/3 |
| Sin casco | ≥32 px | 230/256 | 0.766 → 0.824 | +0.058 | 153/55/22 |

El promedio mejora en todos los grupos de entrenamiento. Esto muestra que el regresor puede ajustar cajas de las imágenes vistas; no garantiza que funcione igual fuera de ellas. Este modo eval sin aumentos difiere de las condiciones históricas de entrenamiento y del replay de muestreo en modo train.

## Validación: muy pocas cabezas por grupo

Hay tres cabezas con casco (una mediana y dos grandes) y seis sin casco (una pequeña, una mediana y cuatro grandes). Cada una se evalúa en ocho modelos. No hay cabezas pequeñas con casco: no se puede estimar ese grupo. «Únicas con propuesta» cuenta referencias que tienen una propuesta elegible en al menos un modelo, no en todos.

| Clase | Tamaño | Cabezas únicas totales / con propuesta | Pares elegibles / total | IoU antes → después | Δ medio | Mejora / empeora | Score ≥0,5 |
|---|---|---|---|---|---:|---|---:|
| Con casco | <16 px | 0 / 0 | — | — | — | — | — |
| Con casco | 16–<32 px | 1 / 1 | 7/8 | 0.606 → 0.637 | +0.032 | 4/2 | 0 |
| Con casco | ≥32 px | 2 / 2 | 13/16 | 0.557 → 0.598 | +0.041 | 9/3 | 2 |
| Sin casco | <16 px | 1 / 1 | 1/8 | 0.433 → 0.361 | -0.071 | 0/1 | 0 |
| Sin casco | 16–<32 px | 1 / 1 | 6/8 | 0.487 → 0.495 | +0.008 | 4/2 | 0 |
| Sin casco | ≥32 px | 4 / 2 | 5/32 | 0.587 → 0.547 | -0.040 | 2/3 | 3 |

En sin casco grande, el promedio es negativo (−0,040), pero solo hay cinco pares elegibles entre 32 posibles. Los 27 ausentes siguen siendo un problema de propuestas, no una medida de calidad del regresor. En sin casco pequeño hay un único par elegible de puntuación muy baja: no se puede concluir que la regresión perjudique sistemáticamente a las cabezas pequeñas.

## Sensibilidad al exigir propuesta inicial con IoU ≥0,50

La propuesta elegida puede cambiar al cambiar el conjunto elegible, por eso este es otro diagnóstico, no una comparación de los mismos pares entre políticas. Dentro de cada política se conserva estrictamente el mismo candidato antes/después.

| Clase / tamaño, val | Pares elegibles | Δ medio | Cruzan de IoU ≥0,5 a <0,5 |
|---|---:|---:|---:|
| Con casco / ≥32 px | 12 | +0.016 | 0 |
| Con casco / 16–<32 px | 6 | +0.017 | 0 |
| Sin casco / ≥32 px | 5 | -0.029 | 1 |
| Sin casco / 16–<32 px | 4 | +0.029 | 1 |
| Sin casco / <16 px | 0 | Sin datos | 0 |

El signo negativo del grupo grande sin casco se mantiene (−0,029), mientras que el grupo pequeño no tiene ninguna propuesta inicial con IoU ≥0,50. Se pierden dos localizaciones inicialmente válidas: una grande sin casco y una mediana sin casco; solo la grande tiene puntuación ≥0,5. Son los casos locales ya identificados, no nuevas pruebas independientes.

## Ejemplos y variación entre modelos

- Image_01474 h0 (sin casco, grande) mejora con los dos controles de la primera semilla, pero empeora con las dos variantes con reserva de esa semilla. El tamaño es el mismo; el efecto depende del modelo y la propuesta seleccionada.
- Image_01087 h0 (sin casco, mediana) cambia de signo según el checkpoint y suele tener puntuación muy baja. Una mejora geométrica no implica una detección aceptada.
- Image_01087 h3 (sin casco, pequeña) tiene una propuesta cercana solo en un checkpoint y con puntuación casi nula. El cuello de botella previo impide evaluar adecuadamente su regresión.

## Decisión antes de otro entrenamiento

No hay evidencia para eliminar la regresión por una regla de tamaño, ni para desactivarla de forma global. En entrenamiento mejora en todos los grupos; en validación su efecto depende de clase, modelo, propuesta y escena, con datos insuficientes para atribuirlo al tamaño por sí solo. La brecha es compatible con generalización insuficiente, pero no identifica una causa única.

Conviene priorizar una revisión de consistencia de cajas de cabeza y más ejemplos humanos independientes que cubran los tamaños y vistas ausentes, antes de cambiar la pérdida de regresión o introducir reglas por tamaño. Mantener el diagnóstico por candidato y el ranking RPN: muchas cabezas ni siquiera llegan al regresor. Si se diseña otro entrenamiento, cambiar una sola variable y conservar controles y ambas semillas; no escoger reglas usando únicamente estos nueve ejemplos de validación. El conteo permanece deshabilitado.

## Evidencia y verificación

- summary.json: estadísticas por grupo, por modelo y por política, metodología y hashes de checkpoints.
- detector-*.json: cada referencia, tamaño efectivo, candidato elegido, puntuación, coordenadas e IoU pareado.
- verification.json: 724 pares cabeza–modelo (652 train, 72 val), pesos y buffers intactos y concordancia de los 72 registros de validación con el diagnóstico anterior para la política ≥0,50.
- 34 pruebas unitarias aprobadas. Ninguna etiqueta, peso o configuración de despliegue cambió.

Reproducción: `.venv/Scripts/python.exe training/analyze_head_regression_size.py`. Rechaza sobrescribir la carpeta; para repetir se necesita un destino nuevo en el script.
