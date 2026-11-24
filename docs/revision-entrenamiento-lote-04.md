# Ampliación de entrenamiento: lote 04

Preparado en `data/annotations/train-separate-v2-04/`: 12 imágenes de 12 grupos nuevos de train, 103 entidades propuestas y 37 vínculos pendientes. Las etiquetas originales incluyen 27 cabezas sin casco y 10 con casco. No son cifras de anotaciones humanas aceptadas todavía.

## Selección

Tres imágenes por criterio: cabezas NoHelmet grandes, NoHelmet pequeñas, escenas con etiquetas mixtas priorizando contextos superpuestos y ejemplos de referencia por hash. Se usan únicamente anotaciones originales y geometría; no se ejecutó el detector para seleccionar casos. Es diversidad por tamaño/contexto, no una certificación automática de diversidad de apariencia.

Se verificaron hashes originales, partición train, grupos distintos y ausencia de imágenes/grupos compartidos con lotes anteriores, incluida la prueba reservada ya evaluada. No se trasladaron imágenes de val/test a train. La separación depende de la calidad de agrupación original.

## Revisión

1. Abrir `review.html` de la carpeta del lote.
2. Revisar cada cabeza y casco; una gorra, cabello o pañuelo no deben clasificarse como casco. Marcar incierto cuando no se pueda distinguir.
3. Dibujar moto y persona separadas. En escenas con varias personas, cada cabeza debe vincularse a su persona, y varias personas pueden compartir moto.
4. Confirmar entidades y relaciones solo cuando sean claras. Si faltan objetos, no marcar cobertura completa y comunicar la imagen y entidad faltante; el editor permite editar propuestas existentes, no crear nuevas.
5. Guardar avances como borrador. Al terminar, introducir identificador y usar «Validar y exportar revisión».

Guardar un solo archivo para las 12 imágenes:

`data/annotations/train-separate-v2-04/train-separate-v2-04-reviewed.json`

No reemplazar `draft.json` ni ningún archivo de los lotes previos. Las imágenes incompletas o ambiguas se excluirán íntegramente al preparar las etiquetas.

## Después de revisar

Validar contra el borrador de este lote y preparar una nueva versión del dataset, preservando los conjuntos anteriores. No se ha iniciado otro entrenamiento ni se han agregado estas propuestas como etiquetas aceptadas.

El resultado del test ya observado no debe convertirse en otra ronda de ajuste de umbrales o selección de modelos sobre esas ocho imágenes. Los próximos cambios se desarrollarán con train/val, y una confirmación final requerirá otro conjunto reservado. La ampliación no garantiza por sí sola resolver la generalización.

Auditoría: `reports/separate-v2/train-batch-04-audit.json`. Borrador, hashes y separación comprobados; sintaxis JavaScript y nombre de exportación correctos; 29 pruebas unitarias aprobadas. No se confirmó ninguna etiqueta en nombre del usuario.
