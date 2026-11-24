# Propuestas y muestreo de seis regiones no-cabeza

Cuatro checkpoints congelados: controles44 y nuevos50, semillas20261124/20261125, last.pt a1950 pasos. Solo las seis escenas revisadas del entrenamiento ampliado. Los controles44 nunca entrenaron con estas escenas; su ejecución aquí sirve como comparación diagnóstica, no como historial de entrenamiento. No se usó ni modificó validación o test.

## Método

Inferencia en eval con RPN2000/800. Replay en modo entrenamiento con RPN4000/2000 y estadísticas de batch; sin aumentos, en tres pares fijos. Veinte sorteos por imagen con la misma semilla de diagnóstico por pareja. Se restauran buffers BN antes de cada pareja y al finalizar. Sin gradientes, optimizer ni cambios de pesos. Se verifica SHA256 del checkpoint y fingerprint completo del estado. La asignación y selección RoI se contrastan con select_training_samples de torchvision.

Región cercana significa IoU≥0.5 con la caja magenta revisada; sensibilidad adicional a0.3. Esto mide solapamiento con la región, no aciertos/FP de toda la imagen ni recall de cabezas. Solo porque una región no es cabeza no se deduce que sea fondo RPN: moto/persona siguen siendo clases de objeto.

## Resultado en las seis regiones (IoU≥0.5)

| Train | Semilla | Sorteos RPN sin negativo cercano /120 | Regiones sin RoI de fondo /6 | RoI fondo seleccionadas, suma de medias | Detecciones finales de cabeza sobre regiones |
|---|---|---:|---:|---:|---|
| 44 | 20261124 | 113/120 | 0/6 | 33.60 | con casco 0.219 |
| 50 | 20261124 | 113/120 | 1/6 | 27.10 | sin casco 0.065 |
| 44 | 20261125 | 111/120 | 3/6 | 27.00 | con casco 0.134, sin casco 0.101 |
| 50 | 20261125 | 109/120 | 2/6 | 21.50 | ninguna |

En ninguna de estas regiones hay detecciones de cabeza con confianza≥0.5. A IoU≥0.5, el modelo50/20261124 mantiene una detección sin casco de0.065 sobre ropa de Image_02208; el modelo50/20261125 no mantiene ninguna. Todos los argmax de las propuestas cercanas de ambos modelos50 son fondo: una clase secundaria aún puede superar el filtro0.05.

## Muestreo por región en los modelos de 50 imágenes

| Imagen | Semilla | Anclas fondo disponibles | Media anclas fondo seleccionadas | RoI fondo disponibles | Media RoI fondo seleccionadas | Sorteos RoI sin fondo cercano /20 |
|---|---|---:|---:|---:|---:|---:|
| Image_00616 | 20261124 | 29 | 0.15 | 7 | 7.00 | 0 |
| Image_00655 | 20261124 | 8 | 0.05 | 0 | 0.00 | 20 |
| Image_02253 | 20261124 | 32 | 0.05 | 3 | 3.00 | 0 |
| Image_00297 | 20261124 | 30 | 0.00 | 1 | 1.00 | 0 |
| Image_02208 | 20261124 | 60 | 0.00 | 12 | 12.00 | 0 |
| Image_02245 | 20261124 | 31 | 0.10 | 7 | 4.10 | 0 |
| Image_00616 | 20261125 | 29 | 0.20 | 2 | 2.00 | 0 |
| Image_00655 | 20261125 | 8 | 0.10 | 0 | 0.00 | 20 |
| Image_02253 | 20261125 | 32 | 0.05 | 0 | 0.00 | 20 |
| Image_00297 | 20261125 | 30 | 0.15 | 1 | 1.00 | 0 |
| Image_02208 | 20261125 | 60 | 0.00 | 12 | 12.00 | 0 |
| Image_02245 | 20261125 | 31 | 0.05 | 10 | 6.50 | 0 |

## Interpretación y sensibilidad

- RPN: existen8–60 anclas de fondo cercanas por región, pero se seleccionan0–0.20 por sorteo. No entró ninguna en113/120 sorteos de la primera semilla y109/120 de la segunda (94.2% y90.8%). No había anclas positivas cercanas aIoU0.5; en Image_02208 hay además55 ignoradas por el matcher.
- RoI: cuando hay propuestas de fondo cercanas, el sampler suele conservar varias; la ausencia de exposición no es uniforme. Image_00655 no tiene ninguna RoI cercana en ambas semillas; Image_02253 tampoco en la segunda. Cero selección en estos casos proviene de falta de propuestas, no de que el sorteo las descarte.
- Con IoU0.3, el RPN omite negativos cercanos en72/120 y65/120 sorteos de los modelos50. La segunda semilla ya tiene propuestas de fondo en las seis regiones y aparecen dos detecciones sin casco de0.075 y0.056. Por tanto, no afirmar que desaparecieron todos los errores alrededor de las regiones.
- A IoU0.3 aparecen algunas RoI asignadas como positivos de objeto; no se deben convertir indiscriminadamente en fondo. El negativo revisado es para cabeza, no una orden de borrar moto/persona. Los JSON conservan las etiquetas asignadas.

## Conclusión

Hay mejora local de clasificación en parte de las regiones, pero exposición desigual. Esto es compatible con que añadir escenas no garantice aprendizaje específico de cada negativo. No demuestra por sí solo la causa de la diferencia en validación: son pesos finales, seis escenas de entrenamiento y un replay sin el orden ni los aumentos históricos. No se registraron las selecciones reales de los1950 pasos y no se pueden reconstruir a partir del checkpoint.

No se recomienda forzar negativos de forma global todavía. El siguiente cambio evaluable sería instrumentar el registro real de exposición por región en un futuro entrenamiento controlado; si se ensaya reservar RoI negativas, usar únicamente candidatos que el matcher ya clasifique como fondo y no se solapen con cabezas reales, manteniendo cuotas y ambas semillas. Las regiones sin propuestas requieren un tratamiento diferente al simple muestreo. Ningún cambio de sampler se aplicó aquí.

Artefactos: protocol.json, comparison.json y JSON por modelo contienen etapas de propuestas, probabilidades, GT solapadas, asignaciones y los20 conteos completos por categoría. Script: training/audit_negative_region_sampling.py; rechaza sobrescribir el destino.
