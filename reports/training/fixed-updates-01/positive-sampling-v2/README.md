# Propuestas y muestreo de regiones positivas

## Resultado principal

En los cuatro checkpoints del paso 1950 existen cajas decodificadas que cubren las seis cabezas sin casco a IoU ≥0,50. El cuello de botella observable está en su ranking de objectness: al elegir las 2000 propuestas previas a NMS, la cobertura cae a 2, 1, 0 y 1 referencias. El filtrado de cajas pequeñas, NMS y el límite posterior no reducen más esa cobertura. Esto no significa que las cajas descartadas se clasificarían bien ni que aumentar top-k baste para resolver la detección.

## Cobertura de las seis cabezas sin casco en validación

| Train | Semilla | Anclas | Decodificadas | Top 2000 | Después de NMS | Propuestas finales |
|---|---|---:|---:|---:|---:|---:|
| 35 | 20261124 | 5/6 | 6/6 | 2/6 | 2/6 | 2/6 |
| 44 | 20261124 | 5/6 | 6/6 | 1/6 | 1/6 | 1/6 |
| 35 | 20261125 | 5/6 | 6/6 | 0/6 | 0/6 | 0/6 |
| 44 | 20261125 | 5/6 | 6/6 | 1/6 | 1/6 | 1/6 |

Se trata de cobertura por referencia, no matching uno a uno. Una misma propuesta puede cubrir más de una cabeza ocluida. La inferencia se ejecutó en batch 2, con idéntica agrupación a la evaluación de los checkpoints; el mAP de cada checkpoint se reprodujo.

### Ranking de la mejor propuesta que cubre cada cabeza

Posición ordenada por objectness antes de top-k; menor es mejor. Un rango >2000 explica el descarte. Se calcula como uno más el número de puntuaciones estrictamente mayores.

| Imagen / cabeza | 35 / semilla 24 | 44 / semilla 24 | 35 / semilla 25 | 44 / semilla 25 |
|---|---:|---:|---:|---:|
| Image_01474 / h0 | 727 | 653 | 3102 | 3947 |
| Image_01087 / h0 | 1235 | 2148 | 2342 | 1594 |
| Image_01087 / h3 | 9689 | 5283 | 23168 | 12477 |
| Image_02284 / h0 | 4284 | 3105 | 4247 | 4187 |
| Image_01756 / h0 | 2558 | 3213 | 3366 | 4022 |
| Image_01756 / h1 | 3506 | 3213 | 4176 | 6928 |

## ¿El muestreo está dejando sin positivos a las cabezas?

La asignación RPN es binaria: objeto/fondo. Para este análisis atribuimos cada ancla positiva a la clase de la caja real asignada. El RPN muestrea hasta 256 anclas por imagen, hasta 50 % positivas. RoI muestrea hasta 512 regiones, hasta 25 % positivas. Son máximos; no garantiza llenar la cuota positiva.

| Train | Semilla | Cabezas sin casco | Positivos RPN medios por cabeza | Fracción de positivos RPN sin casco | Sorteos RPN sin positivo para esa cabeza | Sorteos RoI sin positivo |
|---|---|---:|---:|---:|---:|---:|
| 35 | 20261124 | 43 | 2.82 | 4.29 % | 46/860 | 0/860 |
| 44 | 20261124 | 53 | 2.86 | 3.95 % | 48/1060 | 0/1060 |
| 35 | 20261125 | 43 | 2.78 | 4.22 % | 50/860 | 0/860 |
| 44 | 20261125 | 53 | 2.83 | 3.92 % | 51/1060 | 0/1060 |

Los conteos son la suma/media de 20 sorteos por imagen en una pasada controlada de train. El denominador de sorteos es cabeza × repetición; no son minibatches ni pasos históricos. La mayor parte de los positivos RPN corresponde a motos y personas. La escasez relativa de positivos de cabeza es una hipótesis relevante para probar; no prueba que el muestreo haya causado todos los fallos.

### Efecto de añadir cajas reales al clasificador RoI

| Train | Semilla | Positivos RoI sin casco antes de añadir GT | Media seleccionada después de añadir GT | De ellos, cajas GT añadidas | Cabezas sin ninguna propuesta positiva propia |
|---|---|---:|---:|---:|---:|
| 35 | 20261124 | 312 | 355.0 | 43.0 | 0/43 |
| 44 | 20261124 | 313 | 366.0 | 53.0 | 0/53 |
| 35 | 20261125 | 279 | 322.0 | 43.0 | 0/43 |
| 44 | 20261125 | 276 | 329.0 | 53.0 | 1/53 |

En esta reproducción, cada cabeza sin casco tiene al menos un positivo RoI en los 20 sorteos. En el modelo de 44 imágenes/semilla 25 hay una cabeza que depende de la caja real añadida para obtener un positivo. No hay evidencia aquí de una ausencia general de positivos RoI sin casco que justifique aumentar ciegamente su cuota. Las cajas añadidas son un mecanismo normal de entrenamiento y nunca se añadieron durante inferencia de validación.

## Clasificación de las propuestas que sí sobreviven

- 35 imágenes, semilla 24: las propuestas que cubren Image_01474 h0 e Image_01087 h0 eligen fondo como clase de máxima probabilidad. Una salida de clase sin casco puede aún superar el umbral interno aunque no sea la clase máxima; por eso esto no equivale al número final de TP.
- 44 imágenes, semilla 24: las cuatro propuestas que cubren Image_01474 h0 eligen sin casco; la de mayor IoU asigna aproximadamente 0,94 a esa clase. Es el único caso cubierto por propuestas finales.
- 35 imágenes, semilla 25: no queda ninguna propuesta con IoU ≥0,50 sobre las seis cabezas sin casco.
- 44 imágenes, semilla 25: dos propuestas cubren Image_01087 h0, pero ambas eligen fondo. La de mayor IoU da aproximadamente 0,98 a fondo y 0,02 a sin casco.

Por tanto hay dos fallos observables: ranking de propuestas insuficiente y rechazo como fondo de algunas propuestas válidas. No se puede atribuir todo a NMS, a la cantidad total de imágenes o a la falta de positivos RoI.

## Decisión antes de entrenar otra vez

1. Priorizar un experimento aislado de muestreo de positivos RPN por objeto: reservar, cuando exista, al menos un ancla positiva por caja real y completar el cupo con el muestreo habitual. Mantener el presupuesto total, las etiquetas, el ranking de evaluación y los demás parámetros; comparar ambos controles a pasos fijos y con ambas semillas. Es una hipótesis de mejora, no una corrección demostrada.
2. Registrar durante ese futuro entrenamiento los positivos disponibles y seleccionados por clase/objeto, incluidos los sorteos con cero positivos. La auditoría actual no puede reconstruir las muestras históricas.
3. No cambiar a la vez pérdidas, anclas, top-k y umbrales. Si se estudia un top-k mayor, tratarlo como experimento separado: puede recuperar propuestas, pero no arregla necesariamente su clasificación y aumenta coste.
4. Mantener el conteo deshabilitado. La validación es pequeña y reiteradamente consultada; una confirmación de generalización requerirá una reserva independiente.

## Método y verificación

- Cuatro checkpoints propios last.pt del paso 1950; sin pesos externos, gradientes, optimizador ni entrenamiento nuevo.
- Validación: modo eval y batch 2. Se descompuso el RPN siguiendo la implementación local de torchvision; las propuestas reconstruidas se comprobaron idénticas a filter_proposals.
- Muestreo: pesos finales congelados, modo train incluyendo estadísticas de batch de BatchNorm, orden fijo del manifiesto, sin flip ni aumentos fotométricos. Se restauraron todos los buffers antes de cada minibatch y al finalizar. Veinte sorteos con semillas diagnósticas fijas.
- La selección manual de RoI se contrastó con select_training_samples usando el mismo estado RNG. Se verificaron hashes del dataset, checkpoints y estado completo del modelo antes/después; no cambió ningún peso ni buffer persistente.
- El reparto medido es una reproducción de fin de entrenamiento, no un registro de los 1950 pasos. El orden, las imágenes aumentadas y las estadísticas de batch históricas son distintos. Los resultados no describen toda la trayectoria de aprendizaje.
- Los JSON de esta carpeta conservan cada cabeza, etapa, ranking, probabilidades y conteos. La carpeta positive-sampling anterior es la primera pasada; esta versión añade la separación exacta entre top-k y filtros de tamaño y los rangos de objectness.

Reproducción: `.venv/Scripts/python.exe training/audit_positive_sampling.py --output <carpeta-nueva>`. Rechaza sobrescribir salidas. No usa test.
