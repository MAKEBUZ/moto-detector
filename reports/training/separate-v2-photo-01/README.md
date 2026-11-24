# Experimento de brillo, contraste y desenfoque

Entrenamiento desde cero, sin cargar checkpoints ni pesos externos. 25 imágenes train y las mismas 5 val; 150 épocas, 1950 pasos, semilla 20261124, batch 2, AdamW lr constante 0.0003, misma arquitectura y reflejos horizontales. La huella inicial y el lock del dataset coinciden con el control de 150 épocas.

Único cambio experimental: brillo uniforme 0.85–1.15 y contraste uniforme 0.85–1.15; probabilidad 0.25 de desenfoque gaussiano con radio uniforme 0.2–0.8 píxeles. Se aplica después de reducir a 640 px y antes de la transformación interna del detector, exclusivamente en train. El generador aleatorio fotométrico es independiente para conservar la secuencia de reflejos. Diagnósticos train y validación no llevan aumentos.

Selección sin cambios: mejor mAP global de validación. Control: época 72; fotométrico: época 105. Duración del nuevo bucle: 318.7 segundos.

| AP@0.50 | Control | Fotométrico |
|---|---:|---:|
| motocicleta | 68.98 % | 69.70 % |
| persona_ocupante | 59.98 % | 55.60 % |
| cabeza_con_casco | 0.00 % | 33.66 % |
| cabeza_sin_casco | 0.00 % | 16.83 % |
| Promedio | 32.24 % | 43.95 % |

## Diagnóstico al final, época 150

| Medida | Control train /45 | Nuevo train /45 | Control val /9 | Nuevo val /9 |
|---|---:|---:|---:|---:|
| rpn_covered_iou50 | 38 | 40 | 2 | 2 |
| correct_head_coverage_iou50_score50 | 39 | 39 | 0 | 0 |
| oracle_correct | 44 | 43 | 0 | 0 |

Cobertura por referencia a IoU ≥0.50, no recall con correspondencia uno a uno. `oracle_correct` suministra cajas reales: diagnóstico, no inferencia normal. AP evalúa las salidas normales mediante correspondencia uno a uno. No es exactitud ni COCO mAP50:95.

## Alcance

Conclusión: la política fotométrica mejora el resultado del checkpoint seleccionado: mAP global de 32.24 % a 43.95 % (+11.71 puntos), con AP positivo en ambas clases de cabeza. La clase persona baja de 59.98 % a 55.60 %; no todas las clases mejoran.

La verificación adicional del mejor checkpoint (época 105) encuentra tres de nueve cabezas cubiertas por propuestas RPN y dos con caja final y clase correctas a IoU ≥0.50. Ninguna supera confianza 0.5. El AP se calcula con las salidas normales del detector a partir de su umbral interno 0.05; por eso no equivale a rendimiento operativo con umbral 0.5. La prueba suministrando cajas reales sigue sin aciertos: una caja predicha puede tener características distintas de la caja anotada y obtener la clase correcta con baja puntuación. `best-validation-probe.json` conserva estos resultados.

Se verificaron la reproducción del estado aleatorio inicial, el hash del estado final y la carga del checkpoint seleccionado. Los datos anteriores permanecen intactos. No se ajustó el umbral tras ver estos resultados.

Decisión: conservar esta política como candidata prometedora, no como solución definitiva. El siguiente experimento debe estudiar resolución/características para cabezas pequeñas manteniendo esta referencia y las demás condiciones fijas. Confirmar después en una evaluación revisada independiente y, cuando haya más datos, otras semillas. La validación actual sigue siendo demasiado pequeña y reutilizada para afirmar robustez o habilitar conteo.

Los tres cambios fotométricos se prueban como una política conjunta; este experimento no separa sus efectos individuales. Una sola semilla y cinco imágenes de validación consultadas repetidamente no permiten confirmar generalización. No se usó test, no se cambiaron etiquetas y no se habilitó conteo.

Historial y pérdidas por componente: `history.json`; diagnósticos cada diez épocas: `head-probes/`; comparación y hashes: `comparison.json`. Pesos nuevos en `runs/detector-separate-v2-photo-01/`. El control permanece intacto.

Verificación: 26 pruebas unitarias aprobadas, incluyendo conservación de cajas, etiquetas y RNG de reflejos, y desactivación fotométrica en validación.

Reproducción: training/train.py con --task detector --separate-classes --dataset data/prepared/helmet-separate-v2-expanded-01 --epochs 150 --batch-size 2 --workers 0 --seed 20261124 --constant-lr --head-probe-every 10 --photometric y un --run nuevo.
