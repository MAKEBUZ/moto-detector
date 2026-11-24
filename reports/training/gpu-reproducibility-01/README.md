# Reproducibilidad GPU: misma semilla, dos procesos nuevos

Semilla20261124, dataset50,1950 actualizaciones y3900 presentaciones por proceso. Inicialización aleatoria idéntica; sin pesos preentrenados, reanudación, reserva de positivos/negativos ni cambios al muestreador. Ambos procesos usan los mismos aumentos y registros pasivos. Configuración fijada antes de entrenar en protocol.json. Se mantuvieron las opciones actuales de GPU; no se activó un modo determinista nuevo.

## Primera divergencia observada

| Elemento | Primer paso distinto | Pasos distintos / comparados |
|---|---:|---:|
| image_indices | ninguno | 0/1950 |
| inputs_sha256 | ninguno | 0/1950 |
| targets_sha256 | ninguno | 0/1950 |
| rng_before | 4 | 1947/1950 |
| losses | 2 | 1949/1950 |
| samplers | 2 | 1949/1950 |
| rng_after_forward | 3 | 1948/1950 |
| rng_after_step | 3 | 1948/1950 |
| gradient_sha256 | 1 | 88/88 |
| model_sha256 | 1 | 88/88 |

Entradas, pérdidas, RNG y máscaras se registran en cada paso. Gradientes antes del recorte y estado del modelo se comparan en pasos1–10 y cada25; la tabla distingue primera divergencia observada de pasos no observados. Los RNG incluyen CPU, CUDA, Python y NumPy.

## Magnitud numérica

- Gradientes después del recorte del primer paso: 38 tensores distintos; diferencia absoluta máxima 3.467430361e-06.
- Modelo después del primer paso: 30 tensores distintos; diferencia absoluta máxima 0.0005370602012.
- Modelo final: 54 tensores distintos; diferencia absoluta máxima 2.649129391.

## Resultado final: last.pt, sin elegir mejor época

| Repetición | AP50 sin casco | TP/FP/FN a0.05 | TP/FP/FN a0.5 | mAP50 global |
|---|---:|---|---|---:|
| 1 | 8.42% | 1/1/5 | 0/0/6 | 37.62% |
| 2 | 4.21% | 1/5/5 | 0/0/6 | 24.56% |

AP50 es área bajo precisión/recall aIoU0.5, no exactitud. Validación original de cinco imágenes: seis cabezas sin casco y tres con casco. No se usa test ni se altera validación. Dos repeticiones detectan falta de reproducibilidad pero no estiman su distribución.

## Configuración observada

```json
{
  "deterministic_algorithms": false,
  "cudnn_deterministic": false,
  "cudnn_benchmark": false,
  "cudnn_allow_tf32": true,
  "matmul_allow_tf32": false,
  "cudnn_version": 91900,
  "cuda_version": "12.8",
  "threads": 4
}
```

## Verificación y artefactos

Hashes de inicialización, datos, código relevante y configuración comparados; contador AdamW1950 y3900 presentaciones verificados. La reevaluación reproduce las métricas de ambos checkpoints. Las pruebas de observación pasiva conservan máscaras y RNG; no se modifica el entrenamiento para elegir un resultado favorable.

reproducibility-summary.json detalla diferencias por tensor y por etapa del muestreador. Las trazas completas y los registros de exposición se conservan comprimidos aquí; first-step.pt y last.pt están en los directorios de ejecución. Modelo y gradientes del primer paso permiten separar diferencias en la propagación hacia atrás de diferencias posteriores del optimizador.

Reproducción: training/run_gpu_reproducibility.py; training/report_gpu_reproducibility_models.py; training/report_gpu_reproducibility.py. Usar destinos nuevos; no sobrescribir el protocolo ni ejecuciones.

## Conclusión y siguiente control

La configuración actual no reproduce el entrenamiento bit a bit pese a fijar la semilla. Las imágenes, su orden, aumentos y etiquetas son idénticos en los1950 pasos; los estados CPU/Python/NumPy también coinciden. Los gradientes y el estado del modelo ya difieren en el paso1, con pérdidas y selecciones iniciales iguales. No se identifica aquí un kernel concreto: la primera diferencia observable está en los gradientes, no en los datos ni en el sorteo inicial.

Las asignaciones y máscaras RoI difieren desde el paso2. El RNG CUDA difiere después del forward del paso3; las máscaras negativas RPN difieren desde el paso4 y las positivas desde el5. Las etiquetas RPN y sus cantidades positivas/negativas se mantienen iguales. Esta secuencia es compatible con diferencias numéricas tempranas que cambian las propuestas y, después, el recorrido del muestreo; no demuestra qué operación interna causó la primera diferencia.

Las métricas también varían: AP50 sin casco8.42% frente a4.21%, FP a0.05 de1 frente a5 y mAP global37.62% frente a24.56%. A0.5 ambos omiten las seis cabezas sin casco y no producen predicciones de esa clase. No se promueve ninguno de estos modelos ni se altera el respaldo. El máximo absoluto final de2.649 del estado incluye buffers BatchNorm; no debe interpretarse como diferencia máxima exclusivamente de pesos entrenables.

Antes de cambiar el muestreador, el siguiente control recomendado es configurar explícitamente algoritmos deterministas, detectar las operaciones que no los admitan y repetir una prueba corta de equivalencia GPU antes de dos entrenamientos completos. No se activaron esas opciones durante esta prueba: ambas repeticiones mantuvieron los valores originales. Cambiar determinismo constituirá una nueva condición experimental y necesitará controles nuevos.

Verificación:44 pruebas aprobadas; trazas completas de1950 pasos, código relevante y configuración iguales; inicialización idéntica; AdamW1950 y3900 presentaciones comprobados. Dataset.lock íntegro y validación sin cambios. No se entrenó con pesos preentrenados ni se cambió el muestreador.
