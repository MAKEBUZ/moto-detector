# Reserva de positivos RPN por objeto

Cuatro entrenamientos nuevos desde cero: 35/44 imágenes y semillas 20261124/20261125. Cada uno termina exactamente en 1950 actualizaciones. Se compara last.pt con su control de idéntico dataset y semilla, no con el mejor checkpoint de otra ejecución. No se cargaron pesos externos ni se reanudaron modelos anteriores.

Único cambio configurado de aprendizaje: si el muestreo RPN habitual deja un objeto elegible sin positivos, se sustituye un positivo redundante de otro objeto por uno del objeto omitido. Se conserva la cantidad total de positivos y negativos, así como las anclas negativas seleccionadas. La reparación usa un RNG independiente (semilla de ejecución +1000000), sin consumir el RNG global. No se cambian las etiquetas, anclas, pérdidas, RoI, presupuestos de propuestas, umbrales ni inferencia.

La garantía solo aplica si el objeto tiene al menos un ancla asignada positiva. Si el presupuesto no alcanza para todos los objetos elegibles, se detiene con error; no se excede la cuota ni se oculta el caso. No ocurrió en estas ejecuciones.

| Train | Semilla | AP sin casco control | Con reserva | Diferencia pp | TP/FP/FN a 0,5 control → reserva | mAP control → reserva |
|---|---|---:|---:|---:|---|---|
| 35 | 20261124 | 8.42 % | 16.83 % | +8.42 | 0/0/6 → 0/0/6 | 44.02 % → 37.38 % |
| 44 | 20261124 | 16.83 % | 0.00 % | -16.83 | 1/0/5 → 0/1/6 | 35.77 % → 29.78 % |
| 35 | 20261125 | 0.00 % | 0.00 % | +0.00 | 0/0/6 → 0/0/6 | 35.37 % → 28.11 % |
| 44 | 20261125 | 0.00 % | 8.42 % | +8.42 | 0/1/6 → 1/2/5 | 34.23 % → 28.42 % |

## Selección real durante entrenamiento: sin casco

Los conteos siguientes corresponden a presentaciones de objetos a lo largo de todos los pasos, no a cabezas únicas. «Antes» es el sorteo habitual dentro de esta ejecución justo antes de repararlo; no es una reconstrucción del registro de los controles históricos.

| Train | Semilla | Presentaciones sin casco | Sin ancla positiva disponible | Omitidas antes | Omitidas después | Positivos seleccionados antes → después |
|---|---|---:|---:|---:|---:|---|
| 35 | 20261124 | 4656 | 0 | 290 | 0 | 12938 → 13216 |
| 44 | 20261124 | 4696 | 0 | 263 | 0 | 13284 → 13537 |
| 35 | 20261125 | 4658 | 0 | 260 | 0 | 12994 → 13241 |
| 44 | 20261125 | 4696 | 0 | 264 | 0 | 13282 → 13536 |

## Verificaciones y alcance

Se verificaron: configuración e inicialización iguales al control correspondiente, pesos aleatorios distintos entre semillas, 1950 pasos en los contadores AdamW, mismo número de presentaciones de imágenes por pareja y dataset intacto. La inferencia congelada reproduce las métricas guardadas. Las 34 pruebas incluyen reserva por objeto, presupuesto, negativos, ausencia de positivos, RNG global y equivalencia de inferencia antes/después de instalar el sampler. Una ejecución corta verificó integración y registros.

Los archivos rpn-selections.jsonl.gz contienen, por paso e imagen, identificador de objeto, clase, positivos disponibles, sorteo anterior, selección final e índices de anclas. No guardan imágenes aumentadas ni coordenadas de todas las anclas, y no permiten por sí solos reconstruir toda la entrada aumentada.

La validación sigue teniendo cinco imágenes reutilizadas: seis cabezas sin casco y tres con casco. Dos semillas no bastan para demostrar estabilidad estadística. Se conservaron los controles; test no se usó y el conteo sigue deshabilitado. AP50 no es exactitud ni mAP50:95. El cambio de positivos altera gradientes y trayectoria posterior; mantener semillas no implica trayectorias idénticas.

Reproducción: `.venv/Scripts/python.exe training/run_fixed_updates.py --rpn-object-reserve`. Evaluación: `.venv/Scripts/python.exe training/report_object_reserve.py`. Las salidas existentes se rechazan.

## Interpretación y decisión

La reserva corrigió 1077 omisiones en 18706 presentaciones de cabezas sin casco, sin dejar ninguna elegible sin positivo. Son presentaciones repetidas, no imágenes ni personas distintas. Todos los objetos sin casco tenían al menos un positivo disponible en las entradas aumentadas registradas.

El AP sin casco sube en dos parejas, baja en una y empata en otra; el mAP global baja en las cuatro. A confianza ≥0,5, tres ejecuciones siguen sin aciertos sin casco. La de 44 imágenes/semilla 20261125 obtiene un acierto y dos falsos positivos; la de 44 imágenes/semilla 20261124 pierde el acierto que tenía el control.

Evitar omisiones de objetos en el muestreo funciona como garantía técnica, pero no demuestra una mejora estable de generalización. El flag experimental queda desactivado por defecto. Se conservan los controles y no se habilita conteo.

Siguiente paso recomendado: comparar las etapas de ranking de propuestas y clasificación de estos checkpoints con sus controles, aprovechando los registros reales. No aumentar simultáneamente reserva, top-k o pérdidas buscando una mejora en esta validación pequeña.
