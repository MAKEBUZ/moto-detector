# Prueba de memorización de cabezas desde cero

El detector sí puede aprender una muestra pequeña: alcanzó AP@0.50 de 100 % en ambas clases de cabeza sobre las mismas imágenes utilizadas para entrenar. No es un resultado de validación ni una estimación de generalización.

## Diseño

Cuatro imágenes exclusivamente de train: Image_01306, Image_00051, Image_01737 e Image_01785. Se seleccionaron determinísticamente dos ejemplos por clase priorizando cabezas grandes, sin usar puntuaciones del modelo. Se conservaron todas sus anotaciones de las cuatro clases: hay siete cabezas, dos con casco y cinco sin casco.

Se creó un detector nuevo con cinco salidas y todos los pesos aleatorios. Semilla 20261124; AdamW con lr constante 0.0003 y weight decay 0.0001; batch 2; 300 pasos; sin aumentos ni scheduler. Cada imagen se presentó 150 veces, alternando dos lotes fijos. No se cargó ningún checkpoint anterior ni pesos externos. Se verificaron el lock del dataset, la reproducción de la huella inicial por semilla y la huella final del checkpoint guardado.

El diagnóstico se ejecutó en modo evaluación cada 25 pasos sobre esa misma muestra. Se midieron pérdidas separadas durante entrenamiento, cobertura de propuestas RPN retenidas, coincidencias de anclas y clasificación suministrando cajas reales. Esta última prueba es solo diagnóstica; el AP se calcula con detección normal sin cajas suministradas.

## Evolución

| Paso | Cabezas cubiertas por RPN, IoU ≥0.50 | Cabezas detectadas con clase correcta, IoU ≥0.50 y confianza ≥0.50 | Clase correcta con caja real | mAP@0.50 de cabezas |
|---|---:|---:|---:|---:|
| 0 | 0/7 | 0/7 | 0/7 | 0 % |
| 100 | 0/7 | 0/7 | 6/7 | 0 % |
| 125 | 1/7 | 1/7 | 7/7 | 10.40 % |
| 175 | 5/7 | 5/7 | 7/7 | 80.20 % |
| 250 | 6/7 | 6/7 | 7/7 | 90.10 % |
| 275 | 7/7 | 7/7 | 7/7 | 100 % |
| 300 | 7/7 | 7/7 | 7/7 | 100 % |

La cobertura se calcula por referencia, no como emparejamiento uno a uno. El AP sí usa el evaluador con correspondencia uno a uno. AP@0.50 no es COCO mAP50:95.

| Pérdida | Promedio pasos 1–25 | Promedio pasos 276–300 |
|---|---:|---:|
| Clasificación de regiones | 0.862124 | 0.028192 |
| Regresión de cajas de regiones | 0.722147 | 0.031877 |
| Presencia de objeto en RPN | 0.524779 | 0.006151 |
| Regresión de cajas RPN | 0.040559 | 0.000375 |

Las pérdidas son de todas las clases, no exclusivamente de cabezas. `losses.json` conserva cada paso; `history.json` incluye detalles por cabeza, mejor IoU de ancla y cantidad de anclas positivas con IoU ≥0.50.

## Interpretación y siguiente experimento

El flujo de etiquetas, modelo, propuestas y métrica puede aprender cabezas en este caso controlado. La clasificación con cajas reales aprende antes que la localización por propuestas. No se observa un bloqueo absoluto que obligue a cambiar inmediatamente la arquitectura o las anclas.

La prueba modifica simultáneamente tamaño de muestra, repetición por imagen, aumentos y scheduler respecto al entrenamiento completo. No demuestra cuál de esos factores causó el fallo anterior, ni descarta otros problemas fuera de esta muestra. Conviene ejecutar después un entrenamiento completo desde cero con mayor exposición por imagen, instrumentación de pérdidas/propuestas y un cambio controlado de planificación del aprendizaje. Evaluar periódicamente en la validación original, sin convertir esta muestra memorizada en validación.

No se tocó validación ni test durante esta prueba, ni se habilitó conteo. Los pesos de diagnóstico no deben sustituir al modelo de uso general.

## Artefactos

Pesos: `runs/detector-head-overfit-01/last.pt`. Configuración, pérdidas, resultados y verificación se guardan también en esta carpeta de informes. Este checkpoint tiene formato específico de diagnóstico y no está conectado a `predict.py`.

Reproducción: `python training/overfit_heads.py`. El script rechaza sobrescribir la ejecución existente. Verificación del proyecto: 25 pruebas unitarias aprobadas; inicialización y checkpoint final verificados por hash. No se promete reproducción bit a bit de la trayectoria GPU.
