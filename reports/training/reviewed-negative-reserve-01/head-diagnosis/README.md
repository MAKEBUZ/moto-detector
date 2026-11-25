# Diagnóstico congelado de cabezas sin casco

Comparación de control y reserva de negativos: last.pt a 1950 pasos, modo determinista, eval e inference_mode, lotes de 2 en el orden original. Sin entrenamiento, cambios de pesos, buffers, validación ni uso de test. Hashes de checkpoints y estado verificados antes/después. El postprocesado reconstruido coincide exactamente con la inferencia normal y reproduce el mAP50 guardado.

| Imagen / índice de caja | Mejor IoU RPN control → variante | Propuestas IoU ≥0.5 | Detectada a 0.05 |
|---|---|---|---|
| Image_01474 / 2 | 0.601 → 0.287 | 4 → 0 | True → False |
| Image_01087 / 1 | 0.631 → 0.517 | 2 → 1 | True → True |
| Image_01087 / 6 | 0.029 → 0.259 | 0 → 0 | False → False |
| Image_02284 / 1 | 0.807 → 0.202 | 1 → 0 | True → False |
| Image_01756 / 1 | 0.715 → 0.208 | 2 → 0 | True → False |
| Image_01756 / 3 | 0.406 → 0.180 | 0 → 0 | False → False |

## Hallazgo

Las tres detecciones perdidas (Image_01474/2, Image_02284/1 e Image_01756/1) pierden toda propuesta con IoU ≥0.5. La cobertura RPN cae de 4/6 a 1/6 cabezas. Ninguna de las tres dispone tampoco de una caja ajustada con IoU ≥0.5 y puntuación sin casco >0.05 antes de NMS: no son pérdidas causadas exclusivamente por la supresión final. La única cabeza con propuesta cercana en la variante sigue detectada.

En la propuesta de mayor IoU de cada cabeza perdida, la puntuación de fondo pasa de 0.755 a 1.000, de 0.025 a 0.873 y de 0.408 a 0.919, respectivamente. Las propuestas cambian entre modelos: esto no demuestra por sí solo que el clasificador empeore ante una caja idéntica. Además, fondo dominante no implica descarte automático: el detector conserva clases no-fondo que superan su umbral, aunque no ganen el argmax.

## Decisión

Mantener el control. El primer cuello de botella observado en las tres pérdidas está en las propuestas finales del RPN. Antes de otro entrenamiento conviene localizar la pérdida dentro del RPN (objectness, regresión, ranking o NMS). Un contrafactual con las mismas cajas en ambos clasificadores permitiría separar el efecto de clasificación del cambio geométrico. Este diagnóstico no identifica cuál componente interno del RPN causó la pérdida ni prueba que toda la regresión sea inocua. Una semilla y seis cabezas de validación limitan la conclusión.

diagnosis.json conserva puntuaciones de las cinco clases, IoU antes/después de regresión, candidatos cercanos y supervivencia final. Los índices de caja son de base cero. Script: training/diagnose_negative_reservation_heads.py.
