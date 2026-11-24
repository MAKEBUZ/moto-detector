# Características de mayor resolución: paso 8

Entrenamiento desde cero: 25 imágenes train, las mismas 5 val, 150 épocas / 1950 pasos, batch 2, semilla 20261124, AdamW lr constante 0.0003. Se mantienen brillo y contraste 0.85–1.15 y desenfoque con probabilidad 0.25 y radio 0.2–0.8 px, además de reflejos horizontales. No se cargaron pesos previos ni externos.

Único cambio configurado: stride de la primera convolución del último bloque, de 2 a 1. El backbone produce paso 8 en lugar de 16: doble resolución espacial por eje. Mismos canales, parámetros, valores iniciales aleatorios, tamaños/relaciones de anclas y límites de propuestas. Consecuencias inseparables: más posiciones de anclas y distinto campo receptivo; no se atribuye causalidad exclusivamente a resolución. La entrada sigue limitada a 640 px.

Checkpoint seleccionado por mAP global: control época 105; paso 8 época 96. Duración del nuevo bucle: 304.7 segundos.

| AP@0.50 | Paso 16 + fotométrico | Paso 8 + fotométrico |
|---|---:|---:|
| motocicleta | 69.70 % | 57.43 % |
| persona_ocupante | 55.60 % | 45.64 % |
| cabeza_con_casco | 33.66 % | 33.66 % |
| cabeza_sin_casco | 16.83 % | 16.83 % |
| Promedio | 43.95 % | 38.39 % |

## Diagnóstico de la última época

| Medida | Control train /45 | Paso 8 train /45 | Control val /9 | Paso 8 val /9 |
|---|---:|---:|---:|---:|
| rpn_covered_iou50 | 40 | 21 | 2 | 1 |
| correct_head_coverage_iou50_score50 | 39 | 19 | 0 | 0 |
| oracle_correct | 43 | 44 | 0 | 2 |

Cobertura por caja a IoU ≥0.50, no recall con correspondencia uno a uno. `oracle_correct` suministra la caja real: diagnóstico, no inferencia normal. AP es la evaluación normal a IoU 0.50, no exactitud ni COCO mAP50:95. El AP usa todas las salidas del detector a partir del umbral interno 0.05; el conteo de confianza ≥0.5 se informa por separado.

## Trazabilidad y límites

Conclusión: el cambio a paso 8 no mejora AP de cabezas en el checkpoint seleccionado: ambas clases empatan con el control (33.66 % con casco y 16.83 % sin casco). El mAP global baja de 43.95 % a 38.39 % (-5.56 puntos porcentuales), con deterioro en moto y persona.

La comprobación adicional del mejor checkpoint (época 96) encuentra dos de nueve cabezas con caja final y clase correctas a IoU ≥0.50. Una supera confianza 0.5, frente a cero en el control. Esto es un cambio de confianza/cobertura a un umbral, no una mejora de AP, ni una estimación de precisión por sí sola. La RPN cubre dos de nueve; el clasificador con cajas reales acierta una de nueve. Los detalles están en `best-validation-probe.json`.

En la época final, las propuestas de cabeza de entrenamiento cubren 21/45 frente a 40/45 del control. El mayor mapa mantiene la clasificación con cajas reales relativamente alta (44/45), pero empeora las propuestas retenidas en esta ejecución. La densidad de anclas y el campo receptivo cambiaron; los límites de propuestas se mantuvieron. Esto orienta a inspeccionar puntuaciones y filtrado antes/después de top-k y NMS, pero no demuestra que esos filtros sean la causa.

Decisión: conservar el modelo fotométrico de paso 16 como referencia. No adoptar paso 8 como mejora demostrada ni ajustar varios parámetros a la vez. Si se continúa esta línea, comparar cobertura antes y después del filtrado RPN y luego realizar una sola modificación controlada. Mantener una evaluación independiente pendiente: las cinco imágenes actuales han guiado varios experimentos.

Verificación adicional: se reprodujo el estado inicial aleatorio, se comprobó el estado final guardado y se cargó el mejor checkpoint con arquitectura de paso 8. Registro en `checkpoint-verification.json`. No se inició otro entrenamiento tras esta comparación.

Se verificaron configuración fija, identidad del lock del dataset y valores iniciales. Los pesos finales y checkpoints del control no se sobrescribieron. El argumento feature_stride queda registrado y la inferencia/diagnóstico lo lee para evitar cargar pesos de paso 8 en una arquitectura de paso 16.

27 pruebas unitarias aprobadas: mapa espacial doble, igual número y valores iniciales de parámetros, anclas conservadas, aumento fotométrico solo en train, etiquetas y métricas de cuatro clases. Los registros por época y propuestas se conservan junto a este informe; pesos en runs/detector-separate-v2-stride8-01/.

Una sola semilla, cinco imágenes y nueve cabezas en una validación consultada repetidamente no bastan para demostrar generalización. No se utilizó test ni se habilitó conteo.

Reproducción: training/train.py con --task detector --separate-classes --dataset data/prepared/helmet-separate-v2-expanded-01 --epochs 150 --batch-size 2 --workers 0 --seed 20261124 --constant-lr --head-probe-every 10 --photometric --feature-stride 8 y un --run nuevo.
