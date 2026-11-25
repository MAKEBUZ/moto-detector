# Reserva de negativos revisados: experimento determinista

Dos entrenamientos desde cero comparados a 1950 actualizaciones, semilla 20261124 y 50 imágenes. Control determinista existente frente a variante nueva; sin pesos preentrenados. Checkpoint last.pt, validación original intacta por SHA256 y sin test. Se verificaron inicialización y configuración compartida.

La reserva selecciona un candidato existente de fondo con IoU >=0.5 por región revisada en RPN y RoI. Sustituye un negativo no protegido, conserva los positivos y las cuotas, y no añade propuestas. Es opcional mediante --reserve-reviewed-negatives junto con --log-negative-exposure y --deterministic-gpu.

| Clase | AP50 control → variante | TP/FP/FN a 0.05 | TP/FP/FN a 0.5 |
|---|---|---|---|
| Con casco | 8.42% → 66.34% | 1/5/2 → 2/2/1 | 0/0/3 → 1/0/2 |
| Sin casco | 38.83% → 8.42% | 4/8/2 → 1/5/5 | 1/3/5 → 0/0/6 |

Exposición a negativos cercanos: RPN 32/468 → 468/468; RoI 315/468 → 301/468. La variante realizó 467 sustituciones RPN y 18 RoI. La reserva no puede garantizar exposición RoI si no existe una propuesta cercana etiquetada como fondo.

## Decisión

No adoptar la variante: reduce falsos positivos sin casco, pero pierde tres de cuatro detecciones a confianza 0.05. Mejora con casco, aunque el objetivo conjunto no se cumple. Solo una semilla y cinco imágenes de validación (tres cabezas con casco y seis sin casco); no demuestra generalización. El siguiente diagnóstico debe separar disponibilidad de propuestas y clasificación para localizar la pérdida sin casco antes de otro entrenamiento.

Verificación: 46 pruebas existentes aprobadas y comprobación específica de preservación de positivos/cuota y selección de fondo revisado. comparison.json contiene métricas y exposición; protocol.json fija el experimento. Reproducción de evaluación: training/report_negative_reservation.py.
