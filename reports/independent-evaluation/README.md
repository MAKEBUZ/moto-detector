# Evaluación reservada: primer lote

Estado: revisión humana validada y evaluación completada en las ocho imágenes, sin exclusiones. [Resultados de la evaluación](results-01/README.md): mAP@0.50 de 26.55 %. El protocolo original se conserva intacto como evidencia de las reglas fijadas antes de evaluar. Este lote de test ya no está sin evaluar.

Se seleccionaron 8 imágenes de 8 grupos de la partición test original. Hay 32 entidades propuestas y 11 relaciones pendientes. La selección es determinista por hash entre imágenes elegibles (al menos una cabeza y moto etiquetadas, hasta cinco cabezas y cuatro contextos de moto). No utiliza errores ni puntuaciones del modelo. Se verificaron los hashes originales y ausencia de grupos/hashes compartidos con los borradores previos. Esto no certifica independencia absoluta si los grupos originales son incompletos.

## Modelo y reglas fijadas

`protocol.json` fija el checkpoint `runs/detector-separate-v2-proposals4-01/best.pt` (época 136) mediante SHA256, clases, hash del borrador e inventario. El modelo se reconstruirá con paso 8 y multiplicador de propuestas 4.

- AP@0.50 por clase y promedio, con las salidas normales del detector.
- Precisión y recall por clase a confianza 0.5 e IoU 0.5, correspondencia uno a uno.
- Cobertura RPN de cabezas como diagnóstico separado, no como recall del detector.
- No ajustar umbrales ni seleccionar otro checkpoint en función de este resultado.
- Excluir imágenes incompletas o ambiguas según revisión humana y publicar las exclusiones antes de puntuar.

Es una evaluación pequeña de ocho escenas seleccionadas, no una garantía de funcionamiento en tráfico real. Como exige objetos anotados, no representa escenas negativas. No habilita automáticamente conteo, video ni multas. La asociación cabeza→persona→moto de cuatro clases todavía debe implementarse y evaluarse.

## Qué revisar ahora

Abrir `data/annotations/test-separate-v2-01/review.html`. Completar cajas separadas de moto y persona, revisar cabeza y casco y confirmar relaciones. La caja gris de contexto no es una caja de moto aislada. Marcar cobertura completa solo si no falta ningún objeto pertinente; si falta una entidad, dejarla sin confirmar y comunicar la imagen y objeto para ampliar el lote, pues el editor aún no permite añadir entidades.

Guardar avances con «Guardar borrador JSON». Al terminar, introducir identificador y usar «Validar y exportar revisión». Copiar el archivo como:

`data/annotations/test-separate-v2-01/test-separate-v2-01-reviewed.json`

Un JSON contiene las ocho imágenes. No modificar los anteriores datasets ni trasladar estas escenas a entrenamiento.

## Preparación posterior

```powershell
.venv/Scripts/python.exe -m training.validate_annotation_batch --canonical data/annotations/test-separate-v2-01/draft.json --reviewed data/annotations/test-separate-v2-01/test-separate-v2-01-reviewed.json
.venv/Scripts/python.exe -m training.prepare_separate_labels --canonical data/annotations/test-separate-v2-01/draft.json --reviewed data/annotations/test-separate-v2-01/test-separate-v2-01-reviewed.json --split test --output data/prepared/helmet-separate-v2-test-01
```

Los comandos quedan pendientes de las anotaciones humanas. El exportador conserva la partición test; el cargador de entrenamiento continúa aceptando solo train/val. Tras aceptar las anotaciones, ejecutar la evaluación congelada, registrar resultados y errores, y usar nuevos datos de desarrollo si se requieren cambios posteriores. No seguir afinando sobre esta prueba reservada.

Verificaciones actuales: borrador válido, hashes originales, separación de grupos y archivos, checkpoint fijado, 28 pruebas unitarias aprobadas. No se han confirmado etiquetas en nombre del usuario.
