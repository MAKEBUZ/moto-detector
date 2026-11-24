# Revisión de entrenamiento: lote 05

12 imágenes nuevas de train, de 12 grupos no usados en lotes anteriores. 54 entidades y 19 vínculos propuestos; todas las decisiones siguen pendientes. Las etiquetas de origen no son etiquetas humanas aceptadas. No se usó ningún modelo para seleccionar o etiquetar este lote.

## Abrir y revisar

Abrir `data/annotations/train-separate-v2-05/review.html`. El editor es local y contiene las vistas previas.

| Imagen | Revisar especialmente |
|---|---|
| Image_02333 | Dos cascos con rostro visible; revisar visor, no confundir el rostro con ausencia de casco. |
| Image_00645 | Casco frontal con visor oscuro; verificar clase. |
| Image_00874 | Casco visto desde atrás como contraste frente a cabello. |
| Image_01504 | Ocupantes vistos desde atrás/lateral; separar cabezas y personas. |
| Image_00924 | Vista trasera elevada; revisar cabello y límite de la cabeza. |
| Image_01917 | Dos ocupantes desde atrás; posible oclusión entre cabezas y cuerpos. |
| Image_02129 | Dos ocupantes en vista lateral; separar cajas y vínculos. |
| Image_01829 | Vista trasera oblicua; conductor parcialmente oculto por pasajero. |
| Image_00247 | Dos motos y pasajero parcialmente oculto; verificar cobertura. |
| Image_01491 | Escena distante y oscura; confirmar cabezas pequeñas y todas las motos visibles. |
| Image_01995 | Cabeza desenfocada; marcar incierto si el casco no se puede resolver. |
| Image_01026 | Escena amplia con cabezas pequeñas; verificar objetos faltantes. |

## Criterios

- Casco con visor o rostro visible sigue siendo con casco si se distingue el casco. Cabello, gorra o pañuelo no son casco. Si no se puede distinguir, marcar incierto; no adivinar.
- Dibujar la motocicleta sin incluir al ocupante como parte de la moto; dibujar cada persona por separado. Revisar también las cajas de cabeza originales.
- En oclusiones, distinguir personas y vincular cada cabeza a su persona y moto. No inventar bordes ocultos si no se pueden estimar con el criterio ya utilizado; dejar incierto cuando corresponda.
- Confirmar entidades y relaciones; marcar imagen revisada. Declarar cobertura completa únicamente si todas las motos y personas ocupantes visibles están representadas.
- Si faltan entidades, dejar cobertura incompleta y comunicar imagen y objeto faltante. El editor actual no permite añadir entidades; no reutilizar otra caja para ocultar el faltante.

## Exportación

Introducir tu identificador y usar **Validar y exportar revisión**. Guardar un solo JSON para las 12 imágenes en:

`data/annotations/train-separate-v2-05/train-separate-v2-05-reviewed.json`

No reemplazar `draft.json`. Para avances parciales, usar la exportación de borrador. Al recibir el JSON se validarán cobertura, cajas, clases y vínculos antes de ampliar el dataset. Aún no se ha entrenado ni incorporado este lote como etiquetas aceptadas.

## Selección y alcance

Se inspeccionó una galería determinista con un candidato por cada uno de los 19 grupos elegibles restantes bajo los filtros actuales. Se eligieron 12 grupos visualmente: los motivos son focos de revisión, no clases nuevas ni atributos certificados. Los recortes ampliados de casco permitieron comprobar candidatos con rostro visible y visor oscuro. No todos los cascos tienen visor discernible; el casco trasero se incluye como contraste.

Se excluyeron grupos y hashes de todos los lotes previos, incluyendo val y test. Esta comprobación depende de la agrupación original y no garantiza independencia semántica entre escenas.

Evidencias: `reports/separate-v2/batch-05-selection/selection.json`, galería de candidatos y `reports/separate-v2/train-batch-05-audit.json`.
