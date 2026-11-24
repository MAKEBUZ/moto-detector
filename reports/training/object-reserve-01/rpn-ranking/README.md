# RPN: por qué quedan cabezas sin casco sin propuestas cercanas

## Alcance y resultado

Se revisaron los ocho checkpoints propios del paso 1950 (control/reserva × 35/44 imágenes × dos semillas), sin entrenamiento ni cambios de pesos. Las seis cabezas sin casco de validación tienen al menos un ancla que el matcher asignaría como positiva. Cinco tienen ancla con IoU ≥0,5 y la pequeña restante tiene máximo cercano a 0,42; el matching de baja calidad permite asignarle un positivo. En los ocho modelos, las cajas decodificadas pueden cubrir las seis referencias. No falta por completo una geometría posible.

El problema observable predominante es el ranking: las propuestas que cubren cabezas no reciben prioridad suficiente frente a otras cajas. El top 2000 conserva entre cero y dos cabezas, y NMS no reduce más esa cobertura. La reserva de positivos no corrige uniformemente ese ranking.

## Puntuación de anclas positivas: train frente a val

Para cada cabeza se calcula la puntuación media de sus anclas asignadas positivas; la tabla muestra la mediana de esas medias. Las asignaciones se calculan con el matcher del modelo, usando las etiquetas solo como diagnóstico. Las puntuaciones sigmoid no son probabilidades calibradas ni comparables como exactitud. Train/val tienen distintas escenas y tamaños: esta comparación no controla esos factores.

| Train | Semilla | Modelo | Mediana en train | Mediana en val | Cabezas cubiertas por propuestas finales train | Val |
|---|---|---|---:|---:|---|---|
| 35 | 20261124 | control | 0.970 | 0.275 | 40/43 | 2/6 |
| 35 | 20261124 | reserve | 0.928 | 0.049 | 39/43 | 2/6 |
| 44 | 20261124 | control | 0.934 | 0.221 | 46/53 | 1/6 |
| 44 | 20261124 | reserve | 0.871 | 0.016 | 40/53 | 1/6 |
| 35 | 20261125 | control | 0.966 | 0.097 | 39/43 | 0/6 |
| 35 | 20261125 | reserve | 0.970 | 0.308 | 36/43 | 1/6 |
| 44 | 20261125 | control | 0.838 | 0.105 | 44/53 | 1/6 |
| 44 | 20261125 | reserve | 0.799 | 0.009 | 43/53 | 1/6 |

La puntuación alta en las imágenes vistas y baja en las de validación es compatible con generalización insuficiente del objectness, pero no identifica una causa única. No permite afirmar que basta con más épocas, otro umbral o reservar más positivos.

## Escenarios de ranking, sin cambiar el detector

Cobertura calculada directamente a partir del rango de la mejor caja que cubre cada referencia. Las columnas 4000 y 8000 son escenarios **antes de NMS, límite posterior y clasificación**, no evaluaciones de un detector modificado ni predicciones de recall final. Una misma caja puede cubrir dos cabezas superpuestas.

| Train | Semilla | Modelo | Top 2000 | Top 4000 hipotético | Top 8000 hipotético |
|---|---|---|---:|---:|---:|
| 35 | 20261124 | control | 2/6 | 4/6 | 5/6 |
| 35 | 20261124 | reserve | 2/6 | 3/6 | 5/6 |
| 44 | 20261124 | control | 1/6 | 5/6 | 6/6 |
| 44 | 20261124 | reserve | 1/6 | 4/6 | 5/6 |
| 35 | 20261125 | control | 0/6 | 3/6 | 5/6 |
| 35 | 20261125 | reserve | 1/6 | 3/6 | 6/6 |
| 44 | 20261125 | control | 1/6 | 2/6 | 5/6 |
| 44 | 20261125 | reserve | 1/6 | 3/6 | 5/6 |

Más presupuesto permitiría pasar algunas cabezas al siguiente bloque, pero no asegura que sean clasificadas correctamente, que sus cajas ajustadas conserven IoU ni que no aumenten los falsos positivos o el coste. No se cambió ningún presupuesto operativo en esta revisión.

## Qué ocupa el top 2000

Cada caja decodificada se atribuye a la anotación de mayor IoU si ese IoU alcanza 0,5; en otro caso se registra como bajo solapamiento. Los conteos suman 10000 candidatos (cinco imágenes × 2000) por modelo. Hay redundancia: varias cajas pueden representar el mismo objeto. Bajo solapamiento no significa fondo certificado: puede ser parte de un objeto, una localización mala u otro contenido no anotado.

| Train / semilla / modelo | Moto | Persona | Con casco | Sin casco | Bajo solapamiento |
|---|---:|---:|---:|---:|---:|
| 35 / 20261124 / control | 1361 | 1111 | 17 | 5 | 7506 (75.06 %) |
| 35 / 20261124 / reserve | 1697 | 1396 | 13 | 10 | 6884 (68.84 %) |
| 44 / 20261124 / control | 1477 | 1579 | 5 | 8 | 6931 (69.31 %) |
| 44 / 20261124 / reserve | 2205 | 1742 | 2 | 6 | 6045 (60.45 %) |
| 35 / 20261125 / control | 1270 | 1119 | 8 | 0 | 7603 (76.03 %) |
| 35 / 20261125 / reserve | 1101 | 1291 | 10 | 2 | 7596 (75.96 %) |
| 44 / 20261125 / control | 1498 | 1252 | 20 | 2 | 7228 (72.28 %) |
| 44 / 20261125 / reserve | 1372 | 1183 | 7 | 1 | 7437 (74.37 %) |

No es correcto atribuir todo el descarte a competencia directa entre clases: la mayoría de los candidatos de este corte no alcanza IoU 0,5 con ninguna referencia. La clasificación binaria objeto/fondo del RPN y la calidad geométrica de sus cajas decodificadas no son la misma medida.

## Casos que distinguen los fallos

- **Image_01087 h3, pequeña:** tiene un positivo disponible, pero su mejor caja cubriente queda entre los rangos 5283 y 23168. En varios modelos la puntuación es casi nula. Ampliar a 4000 no la recuperaría en ninguno. Su menor IoU inicial de anclas puede contribuir, sin explicar por sí solo el resultado.
- **Image_02284 h0:** suele tener puntuación de objeto alta, pero el corte de su imagen es todavía mayor. En el control 44/semilla 24, la mejor caja cubriente puntúa 0,893 y queda en rango 3105 porque la puntuación del puesto 2000 es 0,965. Por tanto una puntuación absoluta alta no asegura sobrevivir.
- **Image_01756 h0/h1, vistas traseras superpuestas:** disponen de cuatro y dos anclas positivas. Sus cajas cubrientes quedan por debajo del corte en todos los modelos. No hay ausencia total de supervisión posible; persiste un problema de ordenamiento relativo.
- **Image_01474 h0:** puede sobrevivir con la primera semilla, pero con reserva/44/segunda semilla cae hasta el rango 10261. Es la misma cabeza y geometría de anclas: la variación entre checkpoints es considerable.

## Ancla supervisada frente a caja decodificada

De las 48 combinaciones cabeza sin casco–modelo, la caja cubriente mejor puntuada procede 11 veces de un ancla positiva, 35 veces de una ignorada y 2 veces de una negativa. No son 48 escenas independientes. La etiqueta supervisada se asigna según el ancla original, mientras que aquí la cobertura se mide después de desplazarla. Una caja decodificada buena no tiene necesariamente un score que mida su IoU final. Esto es coherente con el diseño actual y no prueba un error de implementación ni justifica cambiar etiquetas de validación.

## Decisión sobre qué mejorar

La evidencia prioriza estudiar el **ordenamiento y la calidad de las propuestas**, junto con la generalización del objectness. No hay base para tratar todas las cabezas omitidas como un problema exclusivo de tamaños de ancla; cinco de seis ya tienen anclas con IoU ≥0,5. Tampoco se confirma que la reserva mínima por objeto sea suficiente.

Antes de entrenar otra vez, el siguiente experimento útil es congelar estos mismos pesos y comparar el detector completo con top-k previo a NMS 2000 frente a 4000, conservando el límite posterior de 800, anclas, puntuaciones y resto de parámetros. Medir AP, TP/FP/FN, propuestas que sobreviven y tiempo de inferencia de ambos controles y variantes. Se trataría de una prueba diagnóstica explícita, no de adoptar automáticamente un presupuesto mayor ni seleccionar una semilla favorable.

Si las propuestas adicionales llegan pero no producen detecciones, cambiar solo el ranking/presupuesto no bastará: habrá que seguir separando clasificación y regresión. Una futura modificación de entrenamiento dirigida a objectness/calidad debería probarse de forma aislada, con controles y registro por objeto, sin mezclar simultáneamente anclas, muestreo y pérdidas. El conteo permanece deshabilitado.

## Verificación y límites

- Eval mode para train/val, sin aumentos, orden fijo y batch 2. No representa los minibatches históricos aumentados ni las estadísticas de BatchNorm durante entrenamiento.
- Hashes de datasets y checkpoints verificados; fingerprints completos de pesos y buffers intactos.
- Los 72 registros de cabezas de validación coinciden con el diagnóstico anterior en rango y cobertura final.
- La descomposición de propuestas conserva la comprobación exacta contra filter_proposals de torchvision.
- 34 pruebas unitarias aprobadas. Ningún entrenamiento, test, cambio de pesos, umbrales ni despliegue.
- Cinco imágenes de validación reutilizadas y dos semillas: resultados exploratorios, no prueba de generalización.

Archivos: summary.json y detector-*.json contienen asignaciones, puntuaciones, rangos, competencia por imagen y hashes; verification.json registra las comprobaciones. Reproducción: `.venv/Scripts/python.exe training/diagnose_rpn_head_ranking.py`; usar una salida nueva para repetir, pues se rechaza sobrescribir.
