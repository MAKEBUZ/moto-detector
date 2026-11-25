# Comparación congelada: 2000 frente a 4000 propuestas

Único cambio: presupuesto RPN previo a NMS en evaluación. Se conserva el límite posterior de 800, todos los filtros, pesos y buffers. Dos modelos propios, checkpoints last.pt a 1950 actualizaciones. Modo determinista, lotes de 2 y validación original. Sin entrenamiento ni uso de test. Hashes de checkpoints y estado intactos; dataset verificado. Las predicciones con 2000 coinciden exactamente con las guardadas anteriormente.

## Cabezas sin casco

| Modelo | Pre-NMS | AP50 | TP/FP/FN a 0.05 | TP/FP/FN a 0.5 |
|---|---:|---:|---|---|
| control | 2000 | 38.83% | 4/8/2 | 1/3/5 |
| control | 4000 | 38.83% | 4/14/2 | 1/3/5 |
| variant | 2000 | 8.42% | 1/5/5 | 0/0/6 |
| variant | 4000 | 33.66% | 3/12/3 | 2/1/4 |

## Recuperación por cabeza

La variante recupera Image_02284/1 e Image_01756/1 a ambos umbrales. Image_01474/2 recupera cobertura RPN (IoU máximo 0.287 → 0.675), pero no una detección final correcta a 0.05: hace falta revisar clasificación, ajuste RoI y filtrado posterior. La cobertura RPN de la variante pasa de 1/6 a 4/6 y las detecciones de 1/6 a 3/6. En el control, cobertura RPN 4/6 → 5/6 sin nuevas detecciones. Los índices son de base cero.

## Decisión

No activar 4000 por defecto: en el control añade seis falsos positivos sin casco y ningún acierto a 0.05; en la variante añade dos aciertos y siete falsos positivos. A 0.5 la variante sí pasa de cero detecciones a dos aciertos y un falso positivo, un intercambio que merece inspección pero no demuestra generalización. La AP50 del control permanece igual aunque aumentan los FP: AP no describe por sí sola el comportamiento a un umbral fijo.

La variante4000 todavía tiene menos TP y más FP que control2000 a 0.05. Son cinco imágenes de validación con seis cabezas sin casco, una semilla y checkpoints fijos; no se calibraron umbrales ni se eligió otra época. El siguiente diagnóstico puede revisar la cabeza no recuperada y los falsos positivos adicionales, con pesos congelados.

comparison.json incluye las cuatro clases y cobertura por cabeza; archivos de predicciones conservan cajas, puntuaciones y etiquetas. Script: training/compare_negative_reservation_budget.py.
