# Diagnóstico de detecciones faltantes y casco

Se revisaron siete casos problemáticos de cuatro imágenes del piloto humano. Se reutilizan predicciones del detector propio entrenado desde cero; no se modifican pesos, umbral 0.5, IoU 0.5, etiquetas ni particiones. No se utiliza test. Las referencias de casco y cabeza son las revisadas por el usuario. Para moto se conserva el contexto original moto+ocupante, coherente con la clase del detector.

## Resultados por caso

| Imagen / cabeza | Evidencia de las predicciones almacenadas | Diagnóstico |
|---|---|---|
| Image_01575 / h0 | Caja con IoU 0.549 y confianza 0.261, pero clase con casco; la caja sin casco tiene confianza 0.606 e IoU 0.353. | Se combinan mala localización y confusión de clase. Bajar el umbral recuperaría una caja con clase incorrecta. |
| Image_01575 / h1 | Sin casco: confianza 0.825, IoU 0.865. Con casco: confianza 0.243, IoU 0.969. La revisión humana indica con casco. | La cabeza y la moto están localizadas; el detector favorece la clase equivocada. |
| Image_01575 / h2 | Moto lejana: confianza 0.228, IoU 0.815. Cabeza sin casco: confianza 0.681, IoU 0.594; con casco: confianza 0.215, IoU 0.824. | Moto descartada por confianza y clase de casco incorrecta. |
| Image_00135 / h1 | Mejor IoU de cabeza: 0; mejor IoU del contexto de moto: 0.096. | No hay cajas compatibles para ninguno de los dos extremos en las predicciones guardadas. |
| Image_02097 / h0 | Mejor IoU de cabeza: 0.120; contexto de moto: IoU 0.739 y confianza 0.538. | Falta localizar la cabeza, aunque la moto sí está localizada. |
| Image_02097 / h1 | Cabeza sin casco: confianza 0.683, IoU 0.849. Mejor IoU de contexto de moto: 0.300. | Cabeza correcta; falta separar/localizar su moto. |
| Image_02354 / h0 | Con casco: confianza 0.940, IoU 0.912. Sin casco: confianza 0.537, IoU 0.703. Ambas superan el umbral y se superponen suficientemente para activar el conflicto. | Dos clases opuestas sobre la misma cabeza; el sistema se abstiene de asociarla. |

Las puntuaciones son salidas del modelo, no probabilidades calibradas de acierto. Las cifras están redondeadas; `diagnosis.json` conserva su precisión completa.

## Qué cambia respecto al resumen anterior

El evaluador anterior asignaba un motivo principal por vínculo. En Image_00135/h1 registró cabeza no localizada, pero también falta localizar la moto. Este diagnóstico inspecciona ambos extremos independientemente: no deben sumarse sus fallos como si fueran vínculos distintos. Sigue habiendo 5 candidatos correctos de 11 vínculos humanos confirmados; no se ha mejorado el modelo durante esta revisión.

«No hay caja compatible» se refiere a las salidas almacenadas después del filtrado interno y NMS del detector. No permite concluir que la red nunca generó una propuesta interna. Tampoco basta este piloto para identificar la causa del fallo de entrenamiento.

## Inspección visual

[Vista conjunta](contact-sheet.jpg). Verde: cabeza revisada por el usuario. Cian: contexto original moto+ocupante. Rojo: predicción con confianza ≥ 0.5. Amarillo: predicción por debajo de 0.5. Clases del detector: c0=moto+ocupante, c1=cabeza con casco, c2=cabeza sin casco. Se muestran hasta tres predicciones cercanas por extremo, con IoU > 0.05; no todas las detecciones de cada imagen.

Se inspeccionó la vista conjunta. Las escenas incluyen motos superpuestas, un objeto distante con poco contraste y cabezas parcialmente ocluidas. Son condiciones observadas, no causas demostradas del error. No se generaron ni cambiaron anotaciones a partir de esta inspección del asistente.

## Siguiente intervención propuesta

1. Auditar ejemplos de entrenamiento con pañuelos, cascos oscuros, oclusión y motos próximas. Cualquier corrección requiere revisión humana; estas siete imágenes permanecen en validación.
2. Preparar anotaciones separadas de moto/persona en una muestra de entrenamiento, porque el detector actual no identifica personas de forma independiente. No mezclar las cajas nuevas de moto aislada con la clase antigua moto+ocupante.
3. Evaluar de forma controlada mejoras para objetos pequeños y solapados, con entrenamiento desde inicialización aleatoria y comparación en toda la validación. Este piloto sirve para depurar, no para elegir umbrales a partir de siete imágenes.
4. Mantener la abstención ante clases opuestas. Una eventual regla de resolución o uso del clasificador propio debe medirse antes de adoptarse; escoger simplemente la confianza mayor no demuestra seguridad ni precisión global.

El conteo y el video continúan pendientes. No se añadió lógica que convierta estos candidatos en ocupantes confirmados.

## Reproducibilidad

Ejecutar desde la raíz: `.venv/Scripts/python.exe -m training.diagnose_pilot`.

El script valida el JSON humano y comprueba que los hashes de las entradas coincidan con los usados en la comparación anterior. `diagnosis.json` registra hashes, predicciones y resultados. Verificación: 20 pruebas unitarias aprobadas, incluyendo clasificación de fallos por confianza frente a localización y conservación de la abstención por conflicto.
