# Cajas propuestas frente a cajas ajustadas: prueba congelada

## Diseño

Ocho checkpoints propios del paso 1950: cuatro controles y cuatro con reserva de positivos RPN. Se mantiene el mismo peso, entrada, orden de validación y batch 2. Para cada propuesta se calculan los logits una sola vez. Se comparan dos ramas:

- **Ajustadas:** cajas por clase producidas por el regresor original.
- **Propuestas:** para ambas clases de cabeza (con/sin casco) se usan las cajas RPN originales. Las cajas de moto y persona permanecen ajustadas como antes.

Las puntuaciones por candidato propuesta/clase son exactamente iguales entre ramas. Se mantienen corte interno >0,05, filtro de cajas pequeñas, NMS por clase y máximo de 100 detecciones por imagen. Cambiar geometría puede cambiar qué candidatos sobreviven; por tanto las listas finales de puntuaciones no tienen por qué ser idénticas. Esto es parte del efecto medido y no un ajuste de puntuaciones.

La rama normal se contrastó tensor a tensor con torchvision.postprocess_detections y con la salida del modelo completo, y reprodujo el mAP guardado de los ocho checkpoints. Se verificó que pesos, buffers y archivos de checkpoint no cambiaron. No hubo entrenamiento, test ni cambio de despliegue.

## AP sin casco y punto de operación

A es caja ajustada original; P es propuesta RPN sin regresión de cabeza. TP/FP/FN se calculan con correspondencia uno a uno a IoU ≥0,5 y confianza ≥0,5. Hay seis referencias sin casco.

| Train | Semilla | Modelo | AP A | AP P | Diferencia pp | TP/FP/FN A | TP/FP/FN P |
|---|---|---|---:|---:|---:|---|---|
| 35 | 20261124 | control | 8.42 % | 0.00 % | -8.42 | 0/0/6 | 0/0/6 |
| 35 | 20261124 | reserve | 16.83 % | 16.83 % | +0.00 | 0/0/6 | 0/0/6 |
| 44 | 20261124 | control | 16.83 % | 16.83 % | +0.00 | 1/0/5 | 1/2/5 |
| 44 | 20261124 | reserve | 0.00 % | 16.83 % | +16.83 | 0/1/6 | 1/0/5 |
| 35 | 20261125 | control | 0.00 % | 0.00 % | +0.00 | 0/0/6 | 0/0/6 |
| 35 | 20261125 | reserve | 0.00 % | 16.83 % | +16.83 | 0/0/6 | 0/0/6 |
| 44 | 20261125 | control | 0.00 % | 0.00 % | +0.00 | 0/1/6 | 0/1/6 |
| 44 | 20261125 | reserve | 8.42 % | 8.42 % | +0.00 | 1/2/5 | 1/2/5 |

Usar propuestas mejora AP sin casco en 2/8 modelos, empeora en 1/8 y empata en 5/8. No son ocho muestras independientes: comparten las mismas cinco imágenes y derivan de dos semillas. No se usa el promedio como prueba estadística.

## Tres casos que cambian el número de aciertos

1. **Reserva, 44 imágenes, semilla 20261124 — Image_01474 h0:** se recupera un acierto con puntuación idéntica 0,884661. La propuesta tiene IoU 0,577449 y la caja ajustada 0,437982. A confianza 0,5 pasa de 0 TP / 1 FP / 6 FN a 1 TP / 0 FP / 5 FN. Confirma el perjuicio de la regresión en este ejemplo.
2. **Reserva, 35 imágenes, semilla 20261125 — Image_01087 h0:** se recupera un acierto de baja puntuación (0,077859) al usar la propuesta con IoU 0,520302. La caja ajustada queda alrededor de 0,4605. Mejora AP, pero sigue sin aciertos a confianza 0,5.
3. **Control, 35 imágenes, semilla 20261124 — Image_01474 h0:** se pierde el acierto de puntuación 0,406464 que la regresión había llevado a IoU 0,501815. Sin regresión deja de cumplir IoU 0,5. Este caso muestra que el ajuste también puede ser necesario; el resultado está muy cerca de la frontera métrica.

En el control de 44 imágenes/semilla 20261124 se mantiene un acierto a confianza 0,5, pero los falsos positivos suben de 0 a 2. El AP no cambia: el orden de puntuaciones permite que esos errores no reduzcan el AP interpolado, aunque sí empeoren la precisión del punto operativo. No debe evaluarse esta intervención solo por AP.

## Conteos con el umbral interno bajo

A confianza 0,05, para distinguir cambios de ranking de cambios de recuperación.

| Train / semilla / modelo | TP/FP/FN ajustadas | TP/FP/FN propuestas |
|---|---|---|
| 35 / 20261124 / control | 1/3/5 | 0/4/6 |
| 35 / 20261124 / reserve | 1/0/5 | 1/1/5 |
| 44 / 20261124 / control | 1/2/5 | 1/4/5 |
| 44 / 20261124 / reserve | 0/3/6 | 1/2/5 |
| 35 / 20261125 / control | 0/0/6 | 0/0/6 |
| 35 / 20261125 / reserve | 0/1/6 | 1/1/5 |
| 44 / 20261125 / control | 0/6/6 | 0/6/6 |
| 44 / 20261125 / reserve | 1/3/5 | 1/4/5 |

## Efectos en el mAP global

La intervención afecta ambas clases de cabeza, por lo que el mAP global también refleja cambios con casco. No se cambian las coordenadas candidatas de moto/persona; la selección global de un máximo de 100 salidas se conserva. Los JSON incluyen las cuatro clases.

| Train / semilla / modelo | mAP ajustadas | mAP propuestas |
|---|---:|---:|
| 35 / 20261124 / control | 44.02 % | 33.74 % |
| 35 / 20261124 / reserve | 37.38 % | 37.38 % |
| 44 / 20261124 / control | 35.77 % | 35.77 % |
| 44 / 20261124 / reserve | 29.78 % | 33.51 % |
| 35 / 20261125 / control | 35.37 % | 35.37 % |
| 35 / 20261125 / reserve | 28.11 % | 32.32 % |
| 44 / 20261125 / control | 34.23 % | 37.18 % |
| 44 / 20261125 / reserve | 28.42 % | 28.42 % |

## Decisión

**No desactivar la regresión de forma general.** La prueba confirma perjuicios locales, pero también aciertos que dependen de ella y aumentos de falsos positivos sin ella. No se adopta la variante sin regresión como configuración predeterminada, ni se elige retrospectivamente una rama distinta para cada modelo o imagen. El conteo permanece deshabilitado.

El ranking RPN y la clasificación como fondo siguen siendo limitaciones separadas: esta intervención no devuelve propuestas descartadas ni aumenta las puntuaciones. Antes de otro entrenamiento, el siguiente diagnóstico útil es medir el cambio de IoU antes/después de regresión por tamaño y clase sobre train y val, junto con el score de cada región, para identificar si el problema se concentra en cabezas pequeñas, oclusiones o casos de puntuación alta. No variar simultáneamente pérdidas, anclas y umbrales.

La evaluación sigue limitada a cinco imágenes reutilizadas y seis cabezas sin casco. Una mejora en uno o dos casos no demuestra generalización. AP50 no es exactitud ni COCO mAP50:95.

## Archivos y reproducción

- comparison.json: métricas y conteos de las dos ramas para los ocho modelos.
- detector-*.json: predicciones por imagen, referencia humana, índices de propuesta y hash de la matriz de puntuaciones compartida. Coordenadas finales en la escala de entrada del cargador.
- Se comprobó igualdad exacta de scores de cada salida con su candidato original y preservación de cajas candidatas de moto/persona.
- 34 pruebas unitarias aprobadas; además, las comparaciones tensor a tensor y las verificaciones de estado se ejecutaron sobre los ocho checkpoints reales.

Comando: `.venv/Scripts/python.exe training/compare_head_box_regression.py`. Rechaza sobrescribir la carpeta de resultados; para repetir, usar una salida nueva en el script. No modifica el entrenador ni el predictor del proyecto.
