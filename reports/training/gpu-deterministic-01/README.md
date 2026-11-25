# Reproducibilidad GPU: misma semilla, dos procesos nuevos

Semilla20261124, dataset50,1950 actualizaciones y3900 presentaciones por proceso. Inicialización aleatoria idéntica; sin pesos preentrenados, reanudación, reserva de positivos/negativos ni cambios al muestreador. Ambos procesos usan los mismos aumentos y registros pasivos. Configuración fijada antes de entrenar en protocol.json.
Condición nueva: determinismo estricto, TF32 desactivado y RoIAlign de referencia eager; no se compara causalmente su precisión con la condición anterior.

## Primera divergencia observada

| Elemento | Primer paso distinto | Pasos distintos / comparados |
|---|---:|---:|
| image_indices | ninguno | 0/1950 |
| inputs_sha256 | ninguno | 0/1950 |
| targets_sha256 | ninguno | 0/1950 |
| rng_before | ninguno | 0/1950 |
| losses | ninguno | 0/1950 |
| samplers | ninguno | 0/1950 |
| rng_after_forward | ninguno | 0/1950 |
| rng_after_step | ninguno | 0/1950 |
| gradient_sha256 | ninguno | 0/88 |
| model_sha256 | ninguno | 0/88 |

Entradas, pérdidas, RNG y máscaras se registran en cada paso. Gradientes antes del recorte y estado del modelo se comparan en pasos1–10 y cada25; la tabla distingue primera divergencia observada de pasos no observados. Los RNG incluyen CPU, CUDA, Python y NumPy.

## Magnitud numérica

- Gradientes después del recorte del primer paso: 0 tensores distintos; diferencia absoluta máxima 0.
- Modelo después del primer paso: 0 tensores distintos; diferencia absoluta máxima 0.
- Modelo final: 0 tensores distintos; diferencia absoluta máxima 0.

## Resultado final: last.pt, sin elegir mejor época

| Repetición | AP50 sin casco | TP/FP/FN a0.05 | TP/FP/FN a0.5 | mAP50 global |
|---|---:|---|---|---:|
| 1 | 38.83% | 4/8/2 | 1/3/5 | 37.57% |
| 2 | 38.83% | 4/8/2 | 1/3/5 | 37.57% |

AP50 es área bajo precisión/recall aIoU0.5, no exactitud. Validación original de cinco imágenes: seis cabezas sin casco y tres con casco. No se usa test ni se altera validación. Dos repeticiones detectan falta de reproducibilidad pero no estiman su distribución.

## Configuración observada

```json
{
  "deterministic_algorithms": true,
  "cudnn_deterministic": true,
  "cudnn_benchmark": false,
  "cudnn_allow_tf32": false,
  "matmul_allow_tf32": false,
  "cudnn_version": 91900,
  "cuda_version": "12.8",
  "threads": 4,
  "deterministic_warn_only": false,
  "cublas_workspace_config": ":4096:8",
  "pythonhashseed": "20261124",
  "roi_align_implementation": "torchvision_reference_eager"
}
```

## Verificación y artefactos

Hashes de inicialización, datos, código relevante y configuración comparados; contador AdamW1950 y3900 presentaciones verificados. La reevaluación reproduce las métricas de ambos checkpoints. Las pruebas de observación pasiva conservan máscaras y RNG; no se modifica el entrenamiento para elegir un resultado favorable.

reproducibility-summary.json detalla diferencias por tensor y por etapa del muestreador. Las trazas completas y los registros de exposición se conservan comprimidos aquí; first-step.pt y last.pt están en los directorios de ejecución. Modelo y gradientes del primer paso permiten separar diferencias en la propagación hacia atrás de diferencias posteriores del optimizador.

Reproducción: training/run_gpu_reproducibility.py; training/report_gpu_reproducibility_models.py; training/report_gpu_reproducibility.py. Usar destinos nuevos; no sobrescribir el protocolo ni ejecuciones.

## Igualdad completa

Trazas completas iguales: True. Estado AdamW final igual: True. Registros de exposición iguales: True.

Esta prueba comprueba reproducibilidad en esta máquina y versiones instaladas; no garantiza igualdad entre otros dispositivos, versiones ni sistemas operativos. Determinismo no significa mayor precisión ni valida el detector para uso real.

## Decisión y coste

La prueba completa confirma igualdad exacta de las dos ejecuciones en esta instalación. Las predicciones guardadas también coinciden byte a byte. Se conserva el muestreador original; un próximo experimento deberá usar este modo en el control y en la variante. No se promueve el modelo para uso real: persisten falsos positivos y solo hay seis cabezas sin casco en validación.

Duración de entrenamiento: 2109,10 s y 2425,04 s (35,2 y 40,4 minutos). El modo eager es más lento. La nueva condición desactiva TF32 y cambia la implementación de RoIAlign; su diferencia de AP frente al control histórico no puede atribuirse a un cambio de muestreo. Pasaron 46 pruebas; implementation-verification.json conserva las comprobaciones de la implementación.
