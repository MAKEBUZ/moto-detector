# Revisión visual y filtros de inferencia congelados

Se inspeccionaron las tres galerías: 27 falsos positivos adicionales de los ocho modelos con RPN4000. Pesos propios congelados; sin entrenamiento ni modificación de etiquetas, test o inferencia por defecto.

## Hallazgos visuales

- Caso 0, reserva44/20261125, Image_01756, p=0.879: asiento oscuro bajo la pasajera, encima del faro trasero. No es una cabeza omitida. Las dos cabezas reales están arriba y etiquetadas. Tanto la propuesta como la caja ajustada apuntan al asiento.
- Casos 1 y 2: estructuras del fondo junto al conductor de Image_02284.
- Casos 3, 11, 12, 15, 19, 20, 23, 25: regiones de moto, ropa, pierna o asiento; muestran confusión de texturas oscuras con cabeza.
- Casos 4, 9, 13, 17, 21, 24: casco completo o fragmentos del casco/cuello. La categoría automática other_region puede incluir fragmentos de casco con IoU insuficiente: no significa que todos sean fondo visual.
- Casos 6, 7, 8, 14, 22, 26: cajas parciales o excesivamente grandes alrededor de cabezas reales, incluidas las dos cabezas superpuestas de Image_01756.
- Casos 5, 10, 16, 18: contexto y regiones poco nítidas. No se infieren nuevas etiquetas humanas a partir de estos recortes.

Galerías: [1](gallery-1.jpg), [2](gallery-2.jpg), [3](gallery-3.jpg). Rojo: caja ajustada; amarillo: propuesta; verde: cabeza sin casco revisada; cian: casco revisado. Coordenadas en imagen transformada del detector. [Escena del falso positivo de alta confianza](high-confidence.jpg).

## Experimento sin entrenamiento

Se comparó la salida original con tres reglas fijadas antes de ejecutarlas, para RPN2000 y RPN4000, en todos los ocho checkpoints. Reglas aplicadas solo a candidatos de cabeza antes del NMS final:

1. Argmax: la clase de cabeza debe superar las otras cuatro clases, incluido fondo.
2. Persona: centro de la caja dentro del 40% superior de una persona predicha con confianza ≥0.5. No usa personas de referencia.
3. Ambas reglas.

Se conservan geometría y puntuaciones de candidatos aceptados. Los rechazados se filtran; se vuelve a ejecutar NMS y máximo100. Las cifras siguientes suman ocho ejecuciones sobre las mismas seis cabezas sin casco: denominador48, no 48 cabezas independientes. No son resultados de un ensemble ni estimaciones de uso real.

| RPN | Confianza | Regla | TP | FP | FN | Precisión | Recall |
|---|---|---|---|---|---|---|---|
| 2000 | 0.05 | regressed | 4 | 18 | 44 | 18.2% | 8.3% |
| 2000 | 0.05 | argmax | 2 | 5 | 46 | 28.6% | 4.2% |
| 2000 | 0.05 | person | 2 | 6 | 46 | 25.0% | 4.2% |
| 2000 | 0.05 | combined | 1 | 1 | 47 | 50.0% | 2.1% |
| 2000 | 0.5 | regressed | 2 | 4 | 46 | 33.3% | 4.2% |
| 2000 | 0.5 | argmax | 2 | 4 | 46 | 33.3% | 4.2% |
| 2000 | 0.5 | person | 1 | 0 | 47 | 100.0% | 2.1% |
| 2000 | 0.5 | combined | 1 | 0 | 47 | 100.0% | 2.1% |
| 4000 | 0.05 | regressed | 10 | 43 | 38 | 18.9% | 20.8% |
| 4000 | 0.05 | argmax | 4 | 6 | 44 | 40.0% | 8.3% |
| 4000 | 0.05 | person | 7 | 20 | 41 | 25.9% | 14.6% |
| 4000 | 0.05 | combined | 2 | 2 | 46 | 50.0% | 4.2% |
| 4000 | 0.5 | regressed | 3 | 5 | 45 | 37.5% | 6.2% |
| 4000 | 0.5 | argmax | 3 | 5 | 45 | 37.5% | 6.2% |
| 4000 | 0.5 | person | 2 | 1 | 46 | 66.7% | 4.2% |
| 4000 | 0.5 | combined | 2 | 1 | 46 | 66.7% | 4.2% |

La regla argmax no elimina el asiento de p=0.879: sin casco ya era la clase ganadora. La regla persona sí lo elimina, pero también elimina el TP de ese modelo en otra imagen. En RPN4000/confianza0.05, argmax sube la precisión agregada de 18.9% a 40.0%, mientras el recall baja de 20.8% a 8.3%. No debe presentarse ese 40% como un aumento global del acierto. Con ambas reglas a confianza0.5 la precisión alcanza66.7%, pero solo se recupera4.2% de las observaciones de cabezas.

Decisión: no activar ninguno de estos filtros como nuevo comportamiento por defecto. El experimento evidencia un compromiso, no una solución usable. Mantener referencias originales y resultados por semilla. La validación de cinco imágenes se ha reutilizado demasiado para certificar mejoras; hará falta un nuevo conjunto reservado y más ejemplos revisados antes de afirmar generalización.

## Próximo cambio con fundamento

Preparar en entrenamiento (nunca copiando estas cinco imágenes de validación) ejemplos de negativos difíciles: asientos, guardabarros, ropa oscura y fragmentos de cascos. Mantener todas las cabezas reales bien etiquetadas y revisar también negativos para no enseñarlos como fondo por error. La recuperación de cabezas exige además mejorar el ranking del RPN; los filtros finales no recuperan propuestas ausentes. Un siguiente entrenamiento debe comparar ambas semillas, mismos pasos y configuración, y usar pesos aleatorios. No se ejecutó porque la instrucción de esta revisión fue mantener pesos congelados y no entrenar todavía.

## Respaldo

Rama local `codex/backup-best-results-20261124`, commit `bcf424d`. Incluye código, diagnósticos, revisiones y copias completas de los checkpoints best y datasets de los dos resultados relevantes: mAP50 global64.44% (35 imágenes) y AP50 sin casco50.50% (25 imágenes). Son métricas distintas sobre validación; no porcentajes de exactitud de producción. SHA256 de ambas copias verificado en backups/best-results-20261124/manifest.json. No se subió a un remoto. Autor configurado MAKEBUZ, sin coautor adicional. Trabajo nuevo en `codex/frozen-inference-review-20261124`.

## Comprobaciones

Las salidas base reproducen exactamente el modelo y sus métricas anteriores para ambos presupuestos. Hashes de checkpoints y buffers intactos; datasets verificados; no se usaron etiquetas para los filtros. Se guardan métricas de las cuatro clases, conteos, cajas y probabilidades en las carpetas inference-policies e inference-policies-4000. El script reproducible es training/evaluate_inference_policies.py --budget 2000 o --budget 4000; su salida actual usa el sufijo de presupuesto y rechaza sobrescribir carpetas.
