# moto-detector

Proyecto académico de detección de motocicletas y ocupantes, clasificación de casco, conteo y consulta de estadísticas. Todos los modelos se han entrenado desde cero, sin pesos preentrenados.

## Estado

Actualizado al **24 de noviembre de 2026**. Se completaron la depuración de datos, revisiones humanas de cajas y relaciones, entrenamientos desde cero y experimentos de generalización en la RTX 3060. El último experimento de 150 épocas del detector de cuatro clases usa **paso 8, propuestas ×4 y 44 imágenes de entrenamiento**, con mAP@0,50 de validación de **47,26 %**. Sin casco sube de 37,87 % a 43,76 % frente al modelo de 35 imágenes, cuyo mAP global de 64,44 % sigue siendo superior. El conteo, video y asistente de estadísticas siguen pendientes. Un checkpoint anterior ya fue evaluado en el lote test; ese lote no se utilizó en el nuevo entrenamiento ni en esta comparación.

## Resultados acumulados

### Modelos iniciales

| Modelo | Datos de entrenamiento / validación | Resultado de validación |
|---|---|---|
| Clasificador CNN de recortes de cabeza | 2889 / 628 recortes | Macro-F1: 91,00 %; exactitud: 91,72 % |
| Detector inicial de tres clases | 1583 / 347 imágenes | mAP@0,50: 69,12 % |

El detector inicial reconoce **moto con ocupante** como una sola clase, además de cabeza con casco y sin casco. Sus resultados **no son directamente comparables** con los del detector nuevo: cambian las clases, las anotaciones y la validación. El clasificador requiere un recorte de cabeza; no detecta personas por sí solo. [Informe inicial](reports/training/README.md).

### Detector de cuatro clases

Clases: **motocicleta, persona ocupante, cabeza con casco y cabeza sin casco**. El último dataset ampliado tiene 44 imágenes de entrenamiento, 246 cajas y 89 cabezas; la validación contiene 5 imágenes, 25 cajas y 9 cabezas (3 con casco y 6 sin casco). Se conservaron las particiones originales y se excluyeron imágenes con cobertura incompleta o incertidumbre.

Las cifras siguientes corresponden al checkpoint seleccionado por **mAP global de validación**, no al máximo independiente de cada clase.

| Experimento | Train | Épocas | Mejor época | Moto AP | Persona AP | Con casco AP | Sin casco AP | mAP@0,50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Piloto de cuatro clases | 14 | 20 | 18 | 42,57 % | 35,30 % | 0 % | 0 % | 19,47 % |
| Dataset ampliado | 25 | 20 | 4 | 42,57 % | 44,55 % | 0 % | 0 % | 21,78 % |
| Mayor exposición, paso 16 | 25 | 150 | 72 | 68,98 % | 59,98 % | 0 % | 0 % | 32,24 % |
| **Paso 16 + variaciones fotométricas** | **25** | **150** | **105** | **69,70 %** | **55,60 %** | **33,66 %** | **16,83 %** | **43,95 %** |
| Paso 8 + variaciones fotométricas | 25 | 150 | 96 | 57,43 % | 45,64 % | 33,66 % | 16,83 % | 38,39 % |
| Paso 8 + propuestas ×4 | 25 | 150 | 136 | 57,43 % | 58,25 % | 33,66 % | 50,50 % | 49,96 % |
| Lote 04 + paso 8 + propuestas ×4 | 35 | 150 | 108 | 63,54 % | 56,35 % | 100 % | 37,87 % | 64,44 % |
| Lote 05 + paso 8 + propuestas ×4 | 44 | 150 | 75 | 62,38 % | 49,22 % | 33,66 % | 43,76 % | 47,26 % |

Todos estos experimentos usan la misma validación de cinco imágenes. Los de 150 épocas mantienen tasa de aprendizaje constante de 0,0003; los de 20 épocas usaban planificación coseno. La política fotométrica aplica brillo y contraste entre 0,85 y 1,15 y desenfoque suave con probabilidad del 25 %, exclusivamente en entrenamiento. El experimento de paso 8 mantiene esa política y los demás parámetros; duplica la resolución espacial del mapa de características por eje.

**Conclusión actual:** con 44 imágenes mejora el AP sin casco de 37,87 % a 43,76 %, pero el mAP global cae de 64,44 % a 47,26 %. A confianza ≥0,5 sigue habiendo cero aciertos sin casco; desaparece el único falso positivo por encima de ese umbral. A confianza 0,05 se mantienen 3/6 aciertos, con un mejor orden de puntuaciones y 8 falsos positivos frente a 7. No basta para habilitar conteo. Se mantienen 150 épocas, pero las actualizaciones suben de 2700 a 3300; no se aisló el efecto de los datos del número de pasos.

El checkpoint anterior de propuestas ×4 obtuvo mAP@0,50 de 26,55 % en ocho imágenes test, con recall sin casco de 0 % a confianza 0,5. Ese resultado pertenece al modelo anterior; el nuevo no se evaluó en test. [Evaluación independiente](reports/independent-evaluation/results-01/README.md).

### Control a igual número de actualizaciones y dos semillas

Se completaron cuatro entrenamientos nuevos desde cero, cada uno con **1950 actualizaciones exactas**, conservando configuración y validación. Se evaluó `last.pt` del paso fijo, sin elegir la mejor época.

| Semilla | AP sin casco: 35 imágenes | AP sin casco: 44 imágenes | TP/FP/FN a confianza 0,5: 35 → 44 |
|---|---:|---:|---|
| 20261124 | 8,42 % | 16,83 % | 0/0/6 → 1/0/5 |
| 20261125 | 0 % | 0 % | 0/0/6 → 0/1/6 |

**La mejora no se repitió con la segunda semilla; no se considera estable.** No comparar estos resultados de paso fijo con los mejores checkpoints de la tabla anterior. Se igualaron actualizaciones, pero el loader de 35 imágenes tiene un minibatch unitario por época: procesó 3792 imágenes frente a 3900 con el dataset de 44. Dos semillas y cinco imágenes de validación reutilizadas no permiten confirmar generalización. El conteo permanece deshabilitado. [Informe y protocolo](reports/training/fixed-updates-01/README.md).

### Auditoría de propuestas y muestreo (sin entrenamiento nuevo)

En los cuatro checkpoints de paso fijo hay propuestas decodificadas que cubren las seis cabezas sin casco, pero solo 2/6, 1/6, 0/6 y 1/6 sobreviven al top 2000 por objectness; NMS no reduce más esa cobertura. En la reproducción controlada de train, sin casco recibe solo 3,9–4,3 % de los positivos RPN, mientras cada cabeza obtiene positivos RoI en todos los sorteos. Algunas propuestas válidas de validación se clasifican como fondo. Esto prioriza estudiar el muestreo RPN por objeto y registrar sus conteos reales antes de otro cambio. Es diagnóstico con pesos finales, no reconstrucción del muestreo histórico. [Informe detallado](reports/training/fixed-updates-01/positive-sampling-v2/README.md).

### Experimento de reserva de positivos RPN por objeto

Se implementó `--rpn-object-reserve`, desactivado por defecto: conserva el cupo de positivos/negativos y reemplaza positivos redundantes para cubrir objetos elegibles omitidos. Se registran las selecciones reales por paso, imagen y objeto. Se ejecutaron cuatro entrenamientos nuevos desde cero, de 1950 actualizaciones exactas, sin pesos preentrenados.

| Train | Semilla | AP sin casco control | Con reserva | TP/FP/FN con reserva a 0,5 |
|---|---|---:|---:|---|
| 35 | 20261124 | 8,42 % | 16,83 % | 0/0/6 |
| 44 | 20261124 | 16,83 % | 0 % | 0/1/6 |
| 35 | 20261125 | 0 % | 0 % | 0/0/6 |
| 44 | 20261125 | 0 % | 8,42 % | 1/2/5 |

La reserva corrigió **1077 omisiones en 18706 presentaciones de cabezas sin casco**, pero no mejoró consistentemente la detección; el mAP global bajó en las cuatro parejas. No se adopta como nueva referencia ni se habilita conteo. Se verificaron 34 pruebas, integración, presupuestos, contadores AdamW, inicializaciones y reproducción de métricas. [Informe y registros](reports/training/object-reserve-01/README.md).

### Comparación de propuestas tras reservar positivos

La cobertura RPN de las seis cabezas sin casco cambia de 2→2, 1→1, 0→1 y 1→1 entre controles y reserva. El ranking sigue descartando la mayoría de las cabezas; algunas propuestas válidas se clasifican como fondo. En un caso, una propuesta con puntuación sin casco 0,885 pasa de IoU 0,577 a 0,438 tras ajustar la caja y pierde el acierto. La reserva solo incrementó el volumen total de positivos sin casco un 1,9–2,1 % dentro de esas ejecuciones. No se entrenó ni modificó ningún peso para esta revisión. [Informe con ejemplo visual](reports/training/object-reserve-01/proposal-comparison-v2/README.md).

### Prueba congelada: cajas de cabeza sin regresión

Se compararon los ocho checkpoints usando las mismas puntuaciones y filtros, reemplazando únicamente las cajas ajustadas de las dos clases de cabeza por sus propuestas RPN. El AP sin casco mejoró en 2/8 modelos, empeoró en 1/8 y empató en 5/8. Se recuperó el caso con puntuación 0,885 cuya regresión llevaba IoU de 0,577 a 0,438, pero otro control perdió un acierto y otro añadió dos falsos positivos a confianza 0,5. **No se desactiva la regresión de forma general.** No hubo entrenamiento ni cambios de pesos o despliegue. [Resultados completos](reports/training/object-reserve-01/head-box-ablation/README.md).

### IoU de cabezas por tamaño y clase

Se comparó la misma propuesta antes/después del ajuste, con pesos y puntuaciones fijos, sobre train y val de los ocho checkpoints. En entrenamiento el IoU medio mejora en ambas clases y todos los tamaños. En validación, sin casco grande baja de 0,587 a 0,547, pero solo hay cinco pares cabeza–modelo con propuesta cercana entre 32 posibles; el grupo pequeño tiene un único par y puntuación muy baja. No hay evidencia suficiente para desactivar la regresión según tamaño. Se verificaron 724 pares, estados intactos y concordancia con los diagnósticos previos; no se entrenó. [Informe por tamaños y clases](reports/training/object-reserve-01/regression-by-size/README.md).

### Ranking RPN de cabezas sin casco

En los ocho checkpoints, las seis cabezas sin casco de validación tienen anclas asignables como positivas y cajas decodificadas que las cubren. El top 2000 conserva solo 0–2 cabezas. Las puntuaciones de anclas positivas son mucho mayores en train que en val; 60–76 % del top 2000 tiene IoU <0,5 con cualquier anotación (esto no certifica fondo). El cuello de botella observado es la priorización, con variación entre semillas, no una ausencia general de anclas. Se proponen pruebas congeladas de presupuesto antes de otro entrenamiento; no se cambió el detector. [Informe RPN](reports/training/object-reserve-01/rpn-ranking/README.md).

### RPN congelado: 2000 frente a 4000 propuestas

Se compararon los ocho checkpoints cambiando solo el top-k previo a NMS; el límite posterior permaneció en 800. La cobertura de cabezas sin casco aumenta en todos, pero AP mejora en cuatro modelos, baja en uno y empata en tres. A confianza 0,5 solo el control/44/semilla 20261124 gana un acierto (1→2 TP, sin FP); reserva/44/semilla 20261125 añade un FP. Los FP a confianza baja aumentan en los ocho. Se midió tiempo con calentamiento y 20 repeticiones pareadas; hay variabilidad y los datos no equivalen a FPS de video. No se adopta 4000 como configuración global. [Informe de detección y tiempo](reports/training/object-reserve-01/topk-2000-vs-4000/README.md).

### Diagnóstico y asociación

- En el piloto humano de siete imágenes, el sistema propuso correctamente **5 de 11 vínculos confirmados cabeza→moto**. Los demás fallaron por localización o abstención ante clases contradictorias. La comparación usa el contexto antiguo moto+ocupante; no evalúa todavía una cadena completa cabeza→persona→moto.
- En una prueba de memorización con **4 imágenes y 7 cabezas**, el detector alcanzó AP@0,50 del **100 % en ambas clases** sobre la propia muestra entrenada, desde el paso 275. Esto confirma capacidad de aprender esos ejemplos, **no generalización**.
- Tras 150 épocas sin variaciones fotométricas, detectaba 39/45 cabezas de entrenamiento con clase correcta y confianza ≥0,5, frente a 0/9 en validación. Se observó una brecha compatible con sobreajuste.
- La inspección visual mostró fallos tanto en cabezas pequeñas como grandes. En algunos casos la propuesta localiza la cabeza, pero el clasificador la rechaza como fondo. El tamaño no explica por sí solo el problema.

### Límites y decisión

AP@0,50 mide detección a IoU 0,50; **no es exactitud ni COCO mAP@0,50:0,95**. Usa las predicciones del detector desde su umbral interno 0,05, por lo que no equivale al rendimiento con confianza mínima de 0,5. La validación es pequeña y se ha consultado repetidamente para elegir checkpoints; hace falta una evaluación independiente para confirmar cualquier mejora.

Se verificaron inicialización aleatoria, hashes de datos y checkpoints, métricas de cuatro clases y aislamiento de aumentos a entrenamiento. La última batería completó **34 pruebas**. Los pesos propios están en `runs/`, excluido de Git junto con los datasets descargados y preparados; los informes y anotaciones revisadas sí están versionados.

La mejora global todavía no resuelve la detección sin casco. El siguiente análisis es revisar sus fallos en la validación original y ampliar diversidad de entrenamiento; una confirmación independiente requerirá una nueva reserva. No se ha implementado la asociación de cuatro clases ni habilitado conteo, video, asistente de estadísticas o evaluación del casco correctamente abrochado.

## Dataset del último entrenamiento

El lote 05 aportó 9 imágenes completas (3 excluidas por cobertura no declarada). La nueva versión `helmet-separate-v2-expanded-03` contiene **44 imágenes train, 246 cajas y 89 cabezas**; mantiene las mismas 5 imágenes de validación. Se completó un entrenamiento desde cero de 150 épocas sobre esta versión; mejor checkpoint en la época 75. Los modelos anteriores permanecen disponibles. [Detalle y reproducción](docs/dataset-separado-v2-lote-05.md).

## Informes de los experimentos

- [Piloto de cuatro clases](reports/training/separate-v2-pilot-01/README.md)
- [Entrenamiento con dataset ampliado](reports/training/separate-v2-expanded-01/README.md)
- [Prueba de memorización de cabezas](reports/training/head-overfit-01/README.md)
- [Entrenamiento prolongado](reports/training/separate-v2-long-01/README.md)
- [Inspección de cabezas de validación](reports/training/separate-v2-long-01/head-comparison/README.md)
- [Brillo, contraste y desenfoque](reports/training/separate-v2-photo-01/README.md)
- [Características con paso 8](reports/training/separate-v2-stride8-01/README.md)
- [Propuestas ×4](reports/training/separate-v2-proposals4-01/README.md)
- [Ampliación con diez imágenes del lote 04](reports/training/separate-v2-expanded-02/README.md)
- [Entrenamiento con 44 imágenes del lote 05](reports/training/separate-v2-expanded-03/README.md)
- [Revisión de los fallos sin casco](reports/training/separate-v2-expanded-02/nohelmet-review/README.md)
- [Comparación de vínculos revisados](reports/relations-pilot/README.md)
- [Dataset ampliado de cuatro clases](docs/dataset-separado-v2-ampliado.md)

## Documentación de datos y preparación

- [Informe de auditoría](reports/auditoria-dataset.md)
- [Criterios de anotación y preparación](docs/criterios-dataset.md)
- [Resultados automáticos](reports/dataset-audit.json)
- [Recuperación y validación de originales](reports/recuperacion-originales.md)
- [Instrucciones para repetir las descargas](docs/recuperar-originales.md)
- [Fuentes y versiones fijadas](data/sources.lock.json)
- [Resultado de la depuración y particiones](reports/preparation/README.md)
- [Guía del dataset preparado](docs/dataset-preparado.md)
- [Primer entrenamiento y resultados](reports/training/README.md)
- [Entrenadores e inferencia desde cero](training/README.md)
- [Revisión de errores y asociación cabeza–moto](reports/error-review/README.md)

## Reproducir la auditoría

Requiere Git y Node.js, sin paquetes adicionales. Desde la raíz del proyecto:

```powershell
git clone --depth 1 https://github.com/Valery-Rosero/DataSetsIA.git data/sources/DataSetsIA
node scripts/audit-dataset.mjs
powershell -ExecutionPolicy Bypass -File scripts/preview-crops.ps1
```

El JSON registra el commit auditado. Para reproducir exactamente un informe, utilizar ese commit de la fuente. La muestra visual se genera en Windows con System.Drawing; no sustituye la revisión de todos los datos.

`data/sources/` está excluido de Git. Los scripts de auditoría no modifican las etiquetas originales. Las copias en `reports/upstream-reference/` se conservan como evidencia del código y los conteos publicados; no se deben ejecutar desde allí.

El repositorio fuente contiene etiquetas y recortes; las imágenes originales ya se recuperaron por separado en `data/sources/originals/`. No ejecutar su unificador sobre la única copia disponible: elimina su salida anterior antes de regenerarla.

### Clasificación y ajuste de las propuestas adicionales con 4000

Diagnóstico congelado de los ocho modelos: las propuestas adicionales aportan 6 TP y 27 FP sin casco a confianza 0.05 (observaciones repetidas entre modelos). En 25 de esos FP la clase dominante es fondo; ninguno tenía IoU sin casco ≥0.5 antes de la regresión. El contrafactual sin ajuste de cajas sin casco no reduce los FP en ningún modelo. A confianza 0.5 aparecen un TP y un FP adicionales. Se mantiene 2000 como referencia, sin entrenamiento ni cambios de pesos. [Desglose, protocolo y limitaciones](reports/training/object-reserve-01/additional-proposals/README.md).

### Revisión visual y filtros congelados

El falso positivo sin casco de confianza 87.9% corresponde al asiento de una moto. Se revisaron los 27 FP adicionales y se compararon filtros de clase ganadora y posición respecto a personas, con RPN2000/4000 y los ocho checkpoints. Reducen FP pero pierden demasiados TP; no se activan por defecto ni se afirma que el detector esté listo para uso real. No hubo entrenamiento. [Informe y galerías](reports/training/object-reserve-01/visual-review/README.md). Respaldo local de los mejores resultados: rama `codex/backup-best-results-20261124`, commit `bcf424d`, checkpoints y datasets incluidos con SHA256.

### Negativos difíciles pendientes de revisión

Preparadas seis imágenes de seis grupos nuevos de entrenamiento: asientos, ropa y espejos/piezas. Un único JSON conserva la revisión de regiones y las cuatro clases habituales. Sin entrenamiento ni cambios de validación. [Instrucciones](data/annotations/train-hard-negatives-v2-01/LEEME.md) · [Selección y trazabilidad](reports/separate-v2/hard-negatives-selection/README.md).

### Negativos difíciles revisados: dataset de 50 imágenes

Las seis imágenes del lote fueron válidas y completas. Nuevo dataset `helmet-separate-v2-expanded-04`: 50 imágenes de entrenamiento y 267 cajas, con las mismas cuatro clases. Se conservan seis regiones no-cabeza como trazabilidad (tres de ropa y tres piezas), sin clases extra. Validación idéntica por SHA256: cinco imágenes y 25 cajas. No hubo entrenamiento. [Preparación y comprobaciones](docs/dataset-separado-v2-negativos-01.md).

### Entrenamiento desde cero: 50 frente a 44 imágenes, dos semillas

Dos entrenamientos nuevos sin pesos preentrenados, comparados con controles44 verificados:1950 actualizaciones y3900 presentaciones por ejecución; mismo detector, aumentos y muestreo original. AP50 sin casco: semilla20261124,16.83% →16.83%; semilla20261125,0% →33.66%. A confianza0.05, TP/FP/FN:1/2/5 →1/5/5 y0/6/6 →2/4/4. A0.5 ambos modelos nuevos producen1 TP,0 FP y5 FN. No demuestra mejora consistente ni preparación para uso real; también empeora con casco. Sin cambios en validación, test o modelo por defecto. [Informe, protocolo y métricas](reports/training/hard-negatives-fixed1950-01/README.md).

### Auditoría congelada de negativos difíciles

Comparadas las seis regiones revisadas con controles44/modelos50 y ambas semillas. En replay de20 sorteos por imagen, el RPN no selecciona anclas negativas cercanas (IoU≥0.5) en113/120 y109/120 sorteos de los modelos50. RoI sí muestrea varias donde existen propuestas, pero carece de ellas en una o dos regiones según semilla. Mejora local parcial; no demuestra generalización ni reconstruye los sorteos históricos. Pesos y buffers intactos; no hubo entrenamiento. [Informe y sensibilidad aIoU0.3](reports/training/hard-negatives-fixed1950-01/negative-region-audit/README.md).

### Registro de exposición real durante entrenamiento

Implementado `--log-negative-exposure`: observa las selecciones reales RPN/RoI sin cambiar el muestreador, con regiones transformadas junto a las imágenes. Dos ejecuciones completas desde cero,50 imágenes y1950 pasos, semillas20261124/20261125:936 registros por semilla. Exposición con negativos cercanos (IoU≥0.5): RPN38/468 y26/468; RoI298/468 y250/468. El espejo Image_00655 casi nunca se selecciona. AP50 sin casco final8.42% y0%; no se promueven estos modelos. Se detectó falta de reproducibilidad bit a bit en CUDA incluso al repetir sin registro; las pruebas verifican que el observador conserva selecciones, pérdidas de forward y RNG.43 pruebas pasan. [Informe y registros](reports/training/negative-exposure-02/README.md).

### Reproducibilidad GPU: dos repeticiones con la misma semilla

Dos procesos nuevos con semilla20261124,50 imágenes,1950 actualizaciones y muestreador original confirman falta de reproducibilidad bit a bit en la configuración actual. Entradas, orden, aumentos y etiquetas idénticos durante los1950 pasos; primera diferencia de gradientes/estado en paso1 y pérdidas/RoI en paso2. AP50 sin casco8.42% frente a4.21%; aconfianza0.05,1 TP/1 FP frente a1 TP/5 FP. No se selecciona el resultado favorable. Antes de modificar el muestreo se recomienda un control explícitamente determinista. Sin pesos preentrenados ni cambios de validación. [Informe, protocolo y trazas](reports/training/gpu-reproducibility-01/README.md).
