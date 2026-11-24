# Comparación a 1950 actualizaciones y dos semillas

Cuatro ejecuciones nuevas desde cero, sin pesos preentrenados ni reanudación. Diseño fijado en protocol.json antes de entrenar. Datasets de 35/44 imágenes; semillas 20261124/20261125. Se evalúa siempre last.pt del paso 1950, no best.pt ni la mejor época para sin casco.

Se conservaron AdamW lr 0.0003 constante, batch 2, paso 8, propuestas ×4, aumentos fotométricos y diagnóstico cada diez épocas más el final. Validación original de cinco imágenes con seis cabezas sin casco y tres con casco; sin test. Se verificó el contador AdamW de todos los parámetros en 1950, hashes e inicialización idéntica dentro de cada pareja y distinta entre semillas.

| Semilla | Train | AP moto | AP persona | AP con casco | AP sin casco | mAP | Sin casco TP/FP/FN a 0,5 |
|---|---:|---:|---:|---:|---:|---:|---|
| 20261124 | 35 | 64.02 % | 37.29 % | 66.34 % | 8.42 % | 44.02 % | 0/0/6 |
| 20261124 | 44 | 57.43 % | 35.15 % | 33.66 % | 16.83 % | 35.77 % | 1/0/5 |
| 20261125 | 35 | 61.58 % | 46.25 % | 33.66 % | 0.00 % | 35.37 % | 0/0/6 |
| 20261125 | 44 | 24.80 % | 65.38 % | 46.73 % | 0.00 % | 34.23 % | 0/1/6 |

## Diferencias pareadas, 44 menos 35 imágenes

- Semilla 20261124: AP sin casco +8.42 puntos; aciertos a 0,5: +1.
- Semilla 20261125: AP sin casco +0.00 puntos; aciertos a 0,5: +0.

Promedio descriptivo de la diferencia de AP sin casco: +4.21 puntos. Dos semillas no permiten afirmar significación estadística ni estabilidad general.

## Exposición y límites

| Semilla | Train | Época final (parcial) | Actualizaciones | Imágenes procesadas |
|---|---:|---:|---:|---:|
| 20261124 | 35 | 109 | 1950 | 3792 |
| 20261124 | 44 | 89 | 1950 | 3900 |
| 20261125 | 35 | 109 | 1950 | 3792 |
| 20261125 | 44 | 89 | 1950 | 3900 |

El dataset de 35 imágenes termina en la época 109 parcial y el de 44 en la 89 parcial. El loader de 35 tiene un minibatch de una imagen por época; el de 44 usa siempre dos. Igualar pasos no iguala exactamente exposición, composición de minibatches, frecuencia de evaluación ni recorridos del RNG. No se alteró el sampler para conservar el entrenamiento anterior. Los monitores preservan el RNG de PyTorch.

Estas cifras no deben confundirse con los máximos de validación de los experimentos anteriores: aquí el criterio es paso fijo. La validación es pequeña y reutilizada; no equivale a prueba independiente. AP50 no es exactitud ni COCO mAP50:95. No se ajustaron umbrales ni se habilitó conteo.

Verificación: 29 pruebas unitarias y una ejecución real de un solo paso comprobaron el corte a mitad de época. La inferencia congelada reprodujo el mAP de los cuatro checkpoints. Configuraciones, pérdidas, diagnósticos, predicciones y hashes están junto a este informe.

Reproducción: `.venv/Scripts/python.exe training/run_fixed_updates.py`; usa carpetas nuevas para repetir, porque el script rechaza sobrescribir resultados. Evaluación: `.venv/Scripts/python.exe training/report_fixed_updates.py`.

## Decisión

La mejora de AP sin casco aparece en una de las dos semillas. Con la original, ambos datasets detectan una cabeza a confianza 0,05; con 44 imágenes mejora el ordenamiento y ese acierto supera 0,5. Con la segunda semilla ambos omiten las seis cabezas sin casco, y el modelo de 44 imágenes produce un falso positivo a confianza 0,5. No se demuestra una mejora estable ni se selecciona una semilla favorable para promover el modelo.

Los diagnósticos finales de ambas clases de cabeza muestran una brecha train/val: con 44 imágenes, cobertura de clase correcta y puntuación ≥0,5 de 69/89 y 62/89 en train frente a 1/9 y 1/9 en val, según semilla. Son coberturas por referencia en inferencia individual, no recall uno a uno ni una medida exclusiva de sin casco.

Siguiente paso recomendado: comparar las seis referencias sin casco en los cuatro checkpoints, separando propuestas RPN, clasificación y fondo; auditar además el muestreo de regiones positivas por clase antes de cambiar pérdidas, arquitectura o añadir más épocas. Para un control que iguale también minibatches completos harán falta ejecuciones nuevas de ambos datasets, pues aquí se preservó el loader anterior. El conteo sigue deshabilitado.
