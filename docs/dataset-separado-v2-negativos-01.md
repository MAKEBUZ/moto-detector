# Dataset ampliado con negativos difíciles revisados

El JSON train-hard-negatives-v2-01-reviewed.json pasó la validación de identidad de imágenes, particiones, cajas, relaciones y decisiones sobre regiones sospechosas. Se aceptan las seis escenas completas; no hay exclusiones. Se respetan las categorías del revisor: tres regiones de ropa y tres piezas de moto. No se reemplazan por las categorías sugeridas durante la selección.

Se crea data/prepared/helmet-separate-v2-expanded-04, conservando expanded-03 intacto. Entrenamiento: 50 imágenes, 267 cajas (75 motos, 96 ocupantes, 39 cabezas con casco y 57 sin casco). El lote aporta 7 motos, 7 ocupantes y 7 cabezas (3 con casco, 4 sin casco).

Las seis regiones revisadas como no-cabeza se conservan en reviewed_nonhead_regions para trazabilidad; no son cajas de una quinta clase ni objetivos adicionales del cargador. El entrenamiento futuro podrá aprender de estas regiones mediante su muestreo normal de fondo, pero incorporarlas no garantiza que todas sean muestreadas. No se ha cambiado el muestreador ni entrenado.

Validación: mismas cinco imágenes y 25 cajas; manifiesto, COCO e imágenes idénticos por SHA256. Las 44 filas anteriores de entrenamiento permanecen idénticas. Se verificaron todos los hashes de los datasets anterior y nuevo, ausencia de cruces de grupos y carga de las 55 imágenes con etiquetas 1–4. El conversor rechaza regiones pendientes, inciertas o marcadas como cabeza real incluso si se declara cobertura completa.

Recibos: reports/separate-v2/hard-negatives-selection/reviewed-validation.json, reports/separate-v2/helmet-separate-v2-expanded-04-prepared.json y reports/separate-v2/expanded-04-verification.json. El dataset y su dataset.lock.json contienen la procedencia de los JSON revisados. No se modificó el JSON del usuario.

Próximo experimento: entrenar desde inicialización aleatoria y comparar con el dataset anterior a igual número de actualizaciones y ambas semillas, manteniendo el resto fijo. Medir también falsos positivos y recall: no presentar precisión aislada como mejora global. Esta preparación no ejecuta entrenamiento.
