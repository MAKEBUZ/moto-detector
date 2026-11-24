# Paso 8 con cuatro veces más propuestas

Entrenamiento desde cero con pesos aleatorios. Se mantienen 25 imágenes train, 5 val, 150 épocas/1950 pasos, batch 2, semilla 20261124, AdamW lr constante 0.0003, paso 8 y política fotométrica. Test no se utilizó. No se cargaron checkpoints anteriores ni pesos externos.

Único cambio configurado frente al paso 8 anterior: presupuestos de propuestas RPN multiplicados por cuatro, tanto en entrenamiento como en evaluación. Train: antes/después NMS de 1000/500 a 4000/2000. Evaluación: de 500/200 a 2000/800. Son máximos, no cantidades garantizadas. Se conservan umbrales de NMS, anclas, muestreo de regiones y máximo de 100 detecciones finales.

Se modifican conjuntamente los cuatro límites; el experimento no identifica cuál aporta el efecto. La inicialización, configuración restante y lock de datos fueron comprobados contra el control de paso 8.

Mejor época nueva: 136. Duración: 346.8 segundos. Criterio de selección sin cambios: mAP global de validación.

| AP@0.50 | Paso 16 foto | Paso 8 foto | Paso 8 propuestas ×4 |
|---|---:|---:|---:|
| motocicleta | 69.70 % | 57.43 % | 57.43 % |
| persona_ocupante | 55.60 % | 45.64 % | 58.25 % |
| cabeza_con_casco | 33.66 % | 33.66 % | 33.66 % |
| cabeza_sin_casco | 16.83 % | 16.83 % | 50.50 % |
| Promedio | 43.95 % | 38.39 % | 49.96 % |

## Diagnóstico al final

| Medida | Paso 8 train /45 | Nuevo train /45 | Paso 8 val /9 | Nuevo val /9 |
|---|---:|---:|---:|---:|
| rpn_covered_iou50 | 21 | 39 | 1 | 3 |
| correct_head_coverage_iou50_score50 | 19 | 39 | 0 | 1 |
| oracle_correct | 44 | 45 | 2 | 2 |

Cobertura por referencia a IoU ≥0.50; no recall uno a uno. Clasificación con cajas reales (`oracle_correct`) es diagnóstico, no inferencia normal. AP usa correspondencia uno a uno y salidas desde el umbral interno 0.05; no es exactitud ni COCO mAP50:95.

## Verificación y alcance

Resultado: el checkpoint seleccionado de la época 136 alcanza mAP global de 49.96 %, frente a 38.39 % del paso 8 anterior y 43.95 % de la referencia fotométrica de paso 16. AP con casco permanece en 33.66 %; AP sin casco aumenta de 16.83 % a 50.50 %. La clase moto queda por debajo de paso 16 (57.43 % frente a 69.70 %); no todas las clases mejoran.

Diagnóstico del mejor checkpoint: las propuestas cubren 6/9 cabezas; las detecciones finales con clase correcta cubren 4/9 a IoU ≥0.50, y 2/9 superan confianza 0.5. La clasificación suministrando cajas reales acierta 2/9. Detalles en `best-validation-probe.json`. Los resultados de la última época son inferiores, por lo que no debe presentarse el mejor checkpoint como una mejora estable en todas las épocas.

La mayor cobertura es consistente con que los presupuestos de propuestas contribuían a la limitación, pero el cambio afecta también el aprendizaje y no identifica por sí solo qué filtro elimina cada cabeza. Este resultado es una referencia candidata mejor por mAP observado, no una validación independiente. Antes de continuar ajustando hiperparámetros sobre estas cinco imágenes, priorizar una evaluación nueva, revisada y reservada, y medir falsos positivos y asociación cabeza→persona→moto. El conteo continúa deshabilitado.

Se reprodujo la huella inicial aleatoria y se verificó la huella de pesos finales. Los checkpoints se reconstruyen con paso 8 y multiplicador 4, registrados en configuración. Registro adicional: `checkpoint-verification.json`. No se inició otro experimento después de esta comparación.

28 pruebas unitarias aprobadas, incluyendo límites de propuestas, parámetros iniciales idénticos y umbrales conservados. Configuración y pérdidas por componente están en los JSON; `head-probes/` conserva las mediciones periódicas. Checkpoints en runs/detector-separate-v2-proposals4-01/. Las ejecuciones anteriores permanecen intactas.

Una semilla y cinco imágenes de validación consultadas repetidamente no permiten confirmar generalización. No se cambiaron etiquetas, no se usó test y el conteo sigue deshabilitado.

Reproducción: training/train.py con --task detector --separate-classes --dataset data/prepared/helmet-separate-v2-expanded-01 --epochs 150 --batch-size 2 --workers 0 --seed 20261124 --constant-lr --head-probe-every 10 --photometric --feature-stride 8 --proposal-multiplier 4 y un --run nuevo.
