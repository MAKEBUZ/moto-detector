# Registro real de exposición a negativos difíciles

Dos entrenamientos completos nuevos desde inicialización aleatoria, 50 imágenes, semillas20261124/20261125, 1950 actualizaciones y3900 presentaciones cada uno. Muestreador original; sin reserva de positivos/negativos, sin pesos externos y sin reanudar checkpoints. Se registraron las selecciones que realmente calcularon las pérdidas, después de los aumentos y del cambio de tamaño.

Cada región apareció78 veces por etapa;936 filas por semilla (6 regiones ×78 épocas ×2 etapas). Los índices son locales a los candidatos de esa imagen/actualización, no identificadores persistentes. RPN:1 significa objeto,0 fondo,-1 ignorado. RoI:1 moto,2 persona,3 casco,4 sin casco y0 fondo.

## Exposición real (IoU≥0.5 con región revisada)

| Semilla | Etapa | Presentaciones | Sin candidatos fondo | Con candidatos, ninguno seleccionado | Con fondo seleccionado | Total candidatos fondo seleccionados |
|---|---|---:|---:|---:|---:|---:|
| 20261124 | rpn | 468 | 0 | 430 | 38 | 39 |
| 20261124 | roi | 468 | 170 | 0 | 298 | 1827 |
| 20261125 | rpn | 468 | 0 | 442 | 26 | 28 |
| 20261125 | roi | 468 | 217 | 1 | 250 | 1511 |

## Por región: presentaciones con fondo seleccionado /78 (IoU≥0.5)

| Imagen | Semilla | RPN | RoI | RoI sin candidatos de fondo |
|---|---|---:|---:|---:|
| Image_00616 | 20261124 | 6 | 63 | 15 |
| Image_00655 | 20261124 | 0 | 0 | 78 |
| Image_02253 | 20261124 | 9 | 36 | 42 |
| Image_00297 | 20261124 | 5 | 56 | 22 |
| Image_02208 | 20261124 | 12 | 73 | 5 |
| Image_02245 | 20261124 | 6 | 70 | 8 |
| Image_00616 | 20261125 | 4 | 43 | 35 |
| Image_00655 | 20261125 | 2 | 2 | 76 |
| Image_02253 | 20261125 | 2 | 24 | 54 |
| Image_00297 | 20261125 | 4 | 40 | 38 |
| Image_02208 | 20261125 | 10 | 72 | 6 |
| Image_02245 | 20261125 | 4 | 69 | 8 |

## Modelos finales frente a repeticiones anteriores sin registro

| Semilla | Ejecución | AP50 sin casco | TP/FP/FN a0.05 | TP/FP/FN a0.5 | mAP50 global |
|---|---|---:|---|---|---:|
| 20261124 | anterior sin registro | 16.83% | 1/5/5 | 1/0/5 | 30.60% |
| 20261124 | registro real | 8.42% | 1/6/5 | 0/1/6 | 27.35% |
| 20261125 | anterior sin registro | 33.66% | 2/4/4 | 1/0/5 | 44.23% |
| 20261125 | registro real | 0.00% | 0/5/6 | 0/1/6 | 33.26% |

## Reproducibilidad y límites

La prueba CPU conserva exactamente pérdidas, gradientes, buffers y RNG; la prueba GPU con la misma entrada conserva exactamente pérdidas, RNG CPU/CUDA y máscaras de selección RPN/RoI. Se comprueba también que la metadata sigue el volteo y redimensionamiento sin cambiar aumentos. No se hacen sorteos adicionales ni se alteran las máscaras del muestreador.

Las ejecuciones completas CUDA no son reproducibles bit a bit con esta configuración: dos pruebas sin observador a dos pasos ya difieren. Las pérdidas fueron1.739662 y1.737720; con observador1.736006. No atribuir diferencias frente a checkpoints históricos al registro ni tratar estas repeticiones como experimento causal de una mejora. Se conserva el diagnóstico en ../negative-exposure-01/gpu-equivalence-investigation.json. La primera ejecución larga se interrumpió por esta investigación; sus resultados parciales quedan excluidos. Las dos ejecuciones aquí parten de cero.

La validación permanece intacta; no se usa test. El registro no vuelve a ejecutar el muestreador ni garantiza selección de las regiones. Las cajas revisadas no son necesariamente fondo para moto/persona: se registran también positivos e ignorados. Hay sensibilidad aIoU0.3 y desglose temporal por tercios en exposure-summary.json. Los registros completos comprimidos están junto a este informe.

Reproducción: training/run_negative_exposure.py; training/report_negative_exposure_models.py; training/report_negative_exposure.py. No sobrescribir ejecuciones existentes.

## Hallazgos y decisión

- Se confirma baja exposición RPN real:38/468 (8.12%) y26/468 (5.56%) apariciones con algún negativo cercano aIoU0.5. Siempre había anclas negativas disponibles: el límite aquí es la selección aleatoria entre muchas anclas.
- RoI:298/468 (63.68%) y250/468 (53.42%) apariciones con fondo seleccionado. En170 y217 no había candidatos de fondo cercanos; solamente una presentación de la segunda semilla tuvo candidatos pero ninguno seleccionado. Forzar la reserva del muestreador RoI no resolvería la mayoría de las ausencias.
- Image_00655: ninguna selección cercana en ambas etapas durante78 presentaciones de la primera semilla; en la segunda solo2. El resto de regiones tiene exposición muy desigual. Un candidato puede estar cerca de una región no-cabeza y corresponder a persona/casco: no deben forzarse esos positivos como fondo. Se registran3 y1 selecciones positivas de persona aIoU0.5; aIoU0.3 también2 positivas de casco en la segunda semilla.
- Las repeticiones nuevas no mejoran validación: AP50 sin casco8.42% y0%; aconfianza0.05 TP/FP/FN1/6/5 y0/5/6. A0.5 ambos dan0 TP,1 FP y6 FN. No se cambia el modelo por defecto ni se reemplaza el respaldo.

La baja exposición es ahora una observación histórica de estas ejecuciones, no una inferencia del checkpoint final. No prueba por sí sola la causa de la baja generalización. Antes de atribuir mejoras pequeñas a un cambio de sampler conviene establecer un control de reproducibilidad en GPU, porque incluso repetir la misma semilla sin observador produjo trayectorias distintas. Un futuro cambio de exposición debe distinguir reserva RPN (hay candidatos) de disponibilidad de propuestas RoI (a menudo faltan), conservar positivos legítimos y compararse con controles nuevos y ambas semillas.

Verificación final:43 pruebas aprobadas, incluyendo selección exacta CPU/GPU, pérdidas/RNG, gradientes CPU y metadatos con volteo;936 filas únicas por ejecución,78 apariciones por región/etapa, cuotas y etiquetas coherentes. Se revalidaron locks completos de datasets44 y50, sin cambios de validación. Los cuatro checkpoints de comparación se evaluaron con el mismo procedimiento y reproducen sus métricas guardadas. No se afirma equivalencia bit a bit de gradientes CUDA ni de entrenamientos completos.
