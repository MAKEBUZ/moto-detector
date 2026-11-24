# Entrenamiento prolongado desde cero

25 imágenes de entrenamiento, 5 de validación originales; 150 épocas (1950 actualizaciones). Pesos aleatorios, semilla 20261124, batch 2, AdamW lr constante 0.0003. Se conservaron arquitectura y aumentos. No se cargaron pesos anteriores ni externos. Test no se utilizó.

La huella inicial coincide con la de los experimentos anteriores y el manifiesto de validación permanece idéntico. Se modificaron duración y planificación del aprendizaje respecto a las 20 épocas con coseno; no se atribuye causalidad a un único cambio.

Mejor checkpoint por mAP global: época 72. Primera época con AP de cabeza positivo en validación: 35. Duración: 323.8 segundos.

| AP@0.50 | Anterior, 20 épocas | Prolongado, mejor checkpoint |
|---|---:|---:|
| motocicleta | 42.57 % | 68.98 % |
| persona_ocupante | 44.55 % | 59.98 % |
| cabeza_con_casco | 0.00 % | 0.00 % |
| cabeza_sin_casco | 0.00 % | 0.00 % |
| Promedio | 21.78 % | 32.24 % |

## Seguimiento de cabezas

| Época | RPN train /45 | RPN val /9 | Cabezas con clase correcta val /9, confianza ≥0.5 | Clasificación con caja real val /9 |
|---|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 |
| 10 | 1 | 0 | 0 | 0 |
| 20 | 5 | 0 | 0 | 0 |
| 30 | 20 | 1 | 0 | 1 |
| 40 | 30 | 3 | 0 | 1 |
| 50 | 32 | 2 | 0 | 1 |
| 60 | 31 | 1 | 0 | 1 |
| 70 | 34 | 2 | 0 | 0 |
| 80 | 38 | 1 | 0 | 0 |
| 90 | 39 | 1 | 0 | 0 |
| 100 | 40 | 3 | 0 | 1 |
| 110 | 38 | 3 | 0 | 0 |
| 120 | 40 | 3 | 1 | 1 |
| 130 | 40 | 1 | 0 | 1 |
| 140 | 39 | 3 | 0 | 1 |
| 150 | 38 | 2 | 0 | 0 |

Cobertura a IoU ≥0.50 por referencia; no equivale a recall con emparejamiento uno a uno. La clasificación con caja real es una prueba diagnóstica con etiquetas conocidas, no inferencia normal. El AP usa emparejamiento uno a uno y las salidas normales del detector; no es exactitud ni COCO mAP50:95.

## Pérdidas por componente

| Componente | Época 1 | Época 150 |
|---|---:|---:|
| loss_classifier | 0.699885 | 0.066528 |
| loss_box_reg | 0.263715 | 0.121048 |
| loss_objectness | 0.651060 | 0.004646 |
| loss_rpn_box_reg | 0.057716 | 0.011909 |

Las pérdidas incluyen las cuatro clases, no solo cabezas. Los diagnósticos se guardaron cada diez épocas (además de la primera), con estado aleatorio preservado para no alterar el muestreo de entrenamiento.

## Límites y artefactos

Conclusión: mejora el mAP global, pero no se obtiene una mejora estable de cabezas en el checkpoint seleccionado. Durante la ejecución hubo AP de cabeza positivo en algunas épocas: los máximos individuales fueron 16.83 % con casco y 22.44 % sin casco. Son máximos retrospectivos de clases, no métricas de un mismo checkpoint seleccionado ni resultados finales. No se sustituyó el criterio de selección para favorecerlos.

En la época 150, la cobertura de detecciones de cabeza con clase correcta y confianza ≥0.5 fue 39/45 en entrenamiento y 0/9 en validación. La RPN cubrió 38/45 y 2/9 respectivamente; la regresión posterior puede mejorar cajas y por eso la cobertura final puede superar la de propuestas. La clasificación con cajas reales acertó 44/45 frente a 0/9. Esta brecha es compatible con sobreajuste y diferencias entre los ejemplos, pero por sí sola no identifica su causa.

Siguiente paso recomendado: inspeccionar por cabeza las nueve referencias de validación y sus tamaños, apariencia y propuestas, contrastándolas con entrenamiento sin cambiar sus etiquetas automáticamente. Ampliar diversidad revisada y estudiar una mejora controlada para cabezas pequeñas; evitar continuar aumentando épocas sin ese análisis. No se inició otro experimento.

Verificación final: 25 pruebas aprobadas, estado aleatorio inicial reproducido y checkpoint final verificado por huella. Se registran en `checkpoint-verification.json`.

Cinco imágenes y nueve cabezas de validación son una muestra pequeña, utilizada repetidamente para seleccionar checkpoints. No constituye una evaluación independiente final. No se habilitan conteo, multas ni asociación de cuatro clases.

`history.json` contiene métricas y pérdidas por época; `head-probes/` contiene propuestas y diagnósticos detallados. Los checkpoints permanecen en `runs/detector-separate-v2-long-01/`. `comparison.json` incluye hashes y métricas. Las versiones anteriores permanecen intactas.

Reproducción: ejecutar training/train.py con --task detector --separate-classes --dataset data/prepared/helmet-separate-v2-expanded-01 --epochs 150 --batch-size 2 --workers 0 --seed 20261124 --constant-lr --head-probe-every 10 y un --run nuevo. No hay reanudación implícita.
