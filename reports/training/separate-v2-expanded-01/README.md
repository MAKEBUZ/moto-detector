# Segundo entrenamiento desde cero: dataset ampliado

Completado en 20 épocas, RTX 3060, semilla 20261124, batch 2, AdamW lr 0.0003 y coseno. Mismo modelo, aumentos y configuración que el piloto anterior; entrenamiento aumenta de 14 a 25 imágenes. Se verificó identidad de bytes del manifiesto de validación y la misma huella del estado aleatorio inicial. No se cargaron pesos previos ni externos. No se utilizó test. Duración del bucle: 39.38 segundos.

## Comparación en las mismas cinco imágenes de validación

| Métrica AP@0.50 | Piloto anterior (mejor época 18) | Ampliado (mejor época 4) |
|---|---:|---:|
| Motocicleta | 42.57 % | 42.57 % |
| Persona ocupante | 35.30 % | 44.55 % |
| Cabeza con casco | 0 % | 0 % |
| Cabeza sin casco | 0 % | 0 % |
| Promedio | 19.47 % | 21.78 % |

La mejora global de 2.31 puntos porcentuales procede de personas, no de cabezas. No hubo AP positivo en cabezas en ninguna de las 20 épocas del nuevo entrenamiento. El checkpoint final obtiene mAP@0.50 de 19.06 %. Se conserva el criterio de selección por mAP global; no se cambió para favorecer el resultado.

La comparación mantiene épocas, pero no número de actualizaciones: el entrenamiento anterior tuvo 140 pasos y el ampliado 260. Por tanto, no aísla el efecto del volumen de datos frente al de más pasos. La validación tiene solo nueve cabezas (tres con casco), por lo que no permite afirmar mejora generalizable.

## Diagnóstico adicional

En el mejor checkpoint, ninguna de las nueve cabezas de validación tiene una propuesta RPN retenida con IoU ≥0.50 ni una predicción final de cabeza compatible. El clasificador tampoco acierta al recibir sus cajas reales. En entrenamiento ocurre lo mismo con las 45 cabezas.

También se diagnosticó `last.pt` (época 20) para evitar concluir solo a partir de una selección temprana: la RPN cubre 1/45 cabezas de entrenamiento y 0/9 de validación. Las predicciones finales y la clasificación con cajas reales siguen en cero en ambos conjuntos. No hay mejora demostrada en detección de cabezas.

Estas cifras son cobertura por caja, no AP ni correspondencia uno a uno. Se mide la RPN después de filtrar propuestas; no demuestra que el problema esté exclusivamente en anclas. La prueba de clasificación con cajas conocidas usa verdad de referencia y no representa inferencia real.

## Artefactos y comprobaciones

- Pesos nuevos: `runs/detector-separate-v2-expanded-01/best.pt` y `last.pt`.
- `comparison.json`: métricas comparadas, pasos y hashes de checkpoints.
- `config.json`, `history.json`, `result.json` y `requirements.lock.txt`: trazabilidad de la ejecución.
- `head-diagnostics/diagnosis.json` y `last-head-diagnostics/diagnosis.json`: diagnóstico reproducible de ambos checkpoints.

25 pruebas unitarias aprobadas. Los pesos iniciales coinciden por semilla y los finales difieren de los iniciales. El dataset y los pesos anteriores no se sobrescribieron. La asociación de cuatro clases y el conteo continúan deshabilitados.

## Siguiente experimento recomendado

Antes de seguir acumulando épocas o anotaciones, comprobar si el detector puede aprender unas pocas cabezas de entrenamiento: registrar pérdidas por componente, correspondencias positivas de anclas/propuestas y cobertura por tamaño. Si esa prueba falla, revisar el muestreo y la configuración de propuestas/características con cambios aislados. Cualquier nuevo entrenamiento seguirá inicializándose aleatoriamente. No se ejecutó ese experimento adicional en esta comparación.

Reproducción del entrenamiento (requiere otro nombre de ejecución):

```powershell
.venv/Scripts/python.exe training/train.py --task detector --separate-classes --dataset data/prepared/helmet-separate-v2-expanded-01 --epochs 20 --batch-size 2 --workers 0 --seed 20261124 --run detector-separate-v2-expanded-repeat
```
