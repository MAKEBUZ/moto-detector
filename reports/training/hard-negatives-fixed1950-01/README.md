# Negativos difíciles: 44 frente a 50 imágenes

Dos entrenamientos nuevos desde inicialización aleatoria con 50 imágenes, semillas 20261124 y 20261125. Se reutilizan y verifican los controles de 44 imágenes. Sin pesos externos, reanudación ni uso del test. Protocolo fijado antes de entrenar.

Se evalúa last.pt a 1950 actualizaciones AdamW, con 3900 imágenes procesadas por ejecución. Inicialización idéntica dentro de cada pareja y distinta entre semillas. Batch2, lr0.0003 constante, stride8, propuestas×4 (RPN2000 en evaluación), aumentos fotométricos y muestreo original. No se aplican los filtros experimentales de inferencia.

| Semilla | Imágenes | AP50 sin casco | TP/FP/FN a 0.05 | TP/FP/FN a 0.5 | AP50 con casco | mAP50 global |
|---|---:|---:|---|---|---:|---:|
| 20261124 | 44 | 16.83% | 1/2/5 | 1/0/5 | 33.66% | 35.77% |
| 20261124 | 50 | 16.83% | 1/5/5 | 1/0/5 | 0.00% | 30.60% |
| 20261125 | 44 | 0.00% | 0/6/6 | 0/1/6 | 46.73% | 34.23% |
| 20261125 | 50 | 33.66% | 2/4/4 | 1/0/5 | 33.66% | 44.23% |

TP: cabeza sin casco acertada; FP: falsa detección sin casco; FN: cabeza sin casco omitida. Emparejamiento uno a uno a IoU≥0.5. Cada ejecución evalúa las mismas seis cabezas sin casco y tres con casco.

## Comparación por semilla

- Semilla 20261124: AP50 sin casco +0.00 puntos.
  Confianza 0.05: aciertos 1 → 1; falsos positivos 2 → 5; omisiones 5 → 5.
  Confianza 0.5: aciertos 1 → 1; falsos positivos 0 → 0; omisiones 5 → 5.
- Semilla 20261125: AP50 sin casco +33.66 puntos.
  Confianza 0.05: aciertos 0 → 2; falsos positivos 6 → 4; omisiones 6 → 4.
  Confianza 0.5: aciertos 0 → 1; falsos positivos 1 → 0; omisiones 6 → 5.

## Alcance y verificaciones

Los 44 ejemplos anteriores se mantienen; se agregan seis escenas revisadas con siete cabezas reales y seis regiones no-cabeza confirmadas. Estas regiones no son una quinta clase ni tienen muestreo forzado. El experimento evalúa añadir las escenas completas, no permite atribuir causalmente un resultado solo a las regiones negativas.

La validación conserva manifiesto, COCO e imágenes; se verifican los locks completos. Se comprobó el paso1950 de todos los estados AdamW, configuración, pesos iniciales, 3900 presentaciones y reproducción de las métricas guardadas. Los controles se identifican por SHA256 en protocol.json. El código del detector, cargador, métricas y monitor conserva el hash de los controles; train.py contiene soporte posterior para reserva de positivos, desactivado en estas ejecuciones.

El dataset50 finaliza en época78; dataset44 en época89 parcial. Igual número de pasos y presentaciones no significa igual exposición por imagen (78 frente a 88.64 en promedio), composición de minibatches o frecuencia de evaluación por época.

Cinco imágenes de validación repetidamente inspeccionadas y dos semillas no certifican generalización ni uso real. AP50 no es porcentaje de exactitud. No se escoge best.pt ni una semilla favorable y no se cambia el checkpoint de uso por defecto.

Los JSON contienen AP de las cuatro clases, conteos a ambos umbrales, predicciones por imagen y hashes; las subcarpetas guardan configuración, pérdidas por componente, historial y diagnósticos de propuestas/cabezas.

Reproducción: python training/run_fixed_updates.py --hard-negatives; después python training/report_hard_negative_training.py. Se requieren destinos nuevos para repetir, porque se rechaza sobrescribir ejecuciones o informes.

## Decisión tras los resultados

No hay una mejora consistente en ambas semillas. En 20261124 el AP sin casco empata, los FP a0.05 aumentan de2 a5 y el AP con casco cae de33.66% a0%; en 20261125 el AP sin casco sube0% →33.66%, con dos aciertos frente a ninguno y FP6 →4 a0.05. A0.5 ambos modelos50 tienen1 TP,0 FP y5 FN: precisión100% en una única detección y recall16.7%, insuficiente para afirmar que el sistema sea usable. No se promociona la segunda semilla por ser favorable.

La variación de AP global es35.77% →30.60% en la primera semilla y34.23% →44.23% en la segunda. Añadir escenas con negativos no garantiza su selección durante el entrenamiento. El siguiente diagnóstico debe comprobar las propuestas, puntuaciones y muestreo de las seis regiones revisadas en entrenamiento, junto a las omisiones de validación, con pesos congelados antes de modificar pérdidas o entrenar otra vez. No se necesita modificar las etiquetas de validación ni seleccionar umbrales por estos resultados.

Comprobaciones del código: 40 pruebas pasan; scripts compilados; diff sin errores de espacios. Las cuatro inferencias reproducen las métricas de sus checkpoints. Se conservan los checkpoints originales y la rama de respaldo previa.
