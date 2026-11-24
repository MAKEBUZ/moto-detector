# Propuestas adicionales con 4000: clasificación y regresión

Ocho checkpoints propios de 1950 actualizaciones, congelados; dos tamaños de entrenamiento y dos semillas. Sin entrenamiento, pesos externos ni conjunto de prueba. Misma validación de cinco imágenes (seis cabezas sin casco); las observaciones entre modelos no son independientes.

Se sigue el índice del ancla de origen después del NMS RPN. Ninguna propuesta de 2000 desaparece del RPN con 4000 en estos casos; sus probabilidades se mantienen dentro de tolerancia 1e-5. Sí cambian los supervivientes del NMS final. Las salidas reconstruidas reproducen las predicciones guardadas de la comparación anterior.

| Modelo | Semilla | TP/FP con 2000 | TP/FP con 4000 | TP/FP adicionales que sobreviven | TP/FP con 4000 sin regresión sin casco |
|---|---|---|---|---|---|
| control 35 | 20261124 | 1/3 | 2/8 | 1/5 | 1/10 |
| reserve 35 | 20261124 | 1/0 | 1/3 | 0/3 | 1/4 |
| control 44 | 20261124 | 1/2 | 3/5 | 2/3 | 3/8 |
| reserve 44 | 20261124 | 0/3 | 2/4 | 2/1 | 3/4 |
| control 35 | 20261125 | 0/0 | 0/1 | 0/1 | 0/1 |
| reserve 35 | 20261125 | 0/1 | 0/4 | 0/4 | 0/4 |
| control 44 | 20261125 | 0/6 | 0/12 | 0/7 | 0/13 |
| reserve 44 | 20261125 | 1/3 | 2/6 | 1/3 | 2/10 |

Tabla a confianza ≥0.05 e IoU ≥0.5, con emparejamiento uno a uno. La última columna conserva todas las puntuaciones y sustituye únicamente las cajas ajustadas de sin casco por sus propuestas; vuelve a aplicar filtros, NMS y máximo 100. No es una evaluación sin regresión de todas las clases.

## Hallazgos

- En las ocho ejecuciones aparecen 33 detecciones sin casco procedentes de propuestas adicionales: 6 TP y 27 FP. Son observaciones modelo-imagen, no 33 objetos distintos. El incremento neto es 25 FP porque dos FP anteriores dejan de sobrevivir al posprocesamiento.
- Los 27 FP adicionales: 20 regiones con IoU <0.1 respecto a cualquier cabeza sin casco y sin casco etiquetado con IoU ≥0.5; 6 cajas parcialmente localizadas sobre cabezas sin casco (0.1 ≤ IoU <0.5); 1 confusión con casco (IoU ≥0.5). La categoría otras regiones no certifica fondo ni descarta etiquetas omitidas.
- En 25 de los 27 FP adicionales, la clase de mayor probabilidad es fondo; en uno es casco y en uno sin casco. Faster R-CNN evalúa cada clase no-fondo por separado: una probabilidad sin casco >0.05 puede sobrevivir aunque fondo sea el máximo. Es el comportamiento normal del posprocesamiento, no un cambio de puntuaciones al ampliar el RPN.
- Ninguno de los 27 FP adicionales tenía IoU sin casco ≥0.5 antes del ajuste. Por tanto, en estas detecciones el ajuste no convirtió una propuesta correctamente localizada en FP. Los seis TP adicionales ya tenían IoU ≥0.5 antes del ajuste. Esto describe los supervivientes y no prueba que la regresión sea inocua para el conjunto completo.
- El contrafactual sin regresión no reduce los FP de ninguno de los ocho modelos: quedan iguales en tres y aumentan en cinco. Sí recupera un TP en reserva44/20261124, un fallo ya observado en una propuesta compartida.
- Con confianza ≥0.5 solo hay dos detecciones adicionales: un TP de control44/20261124 en Image_01756 (0.5324; IoU 0.5209 → 0.6336) y un FP de reserva44/20261125 en la misma imagen (0.8790; IoU sin casco 0 → 0). Este último requiere inspección visual específica; no se soluciona suprimiendo el ajuste.

## Decisión

Mantener 2000 como referencia. No desactivar globalmente la regresión. Antes de entrenar, revisar visualmente las regiones con alta puntuación y comprobar etiquetas/recortes; después, un diagnóstico congelado de la competencia fondo/casco/sin casco puede medir una regla de margen o argmax como contrafactual, incluyendo cuántos TP perdería. No elegir umbrales nuevos por esta validación pequeña y reutilizada.

## Verificación

Hashes de checkpoints y parámetros/buffers intactos; dataset.lock verificado; predicciones y conteos de las ocho ejecuciones reproducidos para ambos presupuestos y umbrales. 36 pruebas pasan (incluyen emparejamiento uno a uno y ausencia de GT sin casco). El primer intento se detuvo al encontrar una imagen sin GT sin casco; se corrigió la reducción vacía, se conservó su protocolo en additional-proposals-incomplete y se ejecutó el análisis completo desde el inicio.

Reproducir en una carpeta nueva: `python training/audit_additional_proposals.py` (rechaza sobrescribir el resultado existente). Los JSON por modelo incluyen ancla, probabilidades de las cinco clases, cajas antes/después, IoU y clasificación TP/FP por detección.
