# Primer lote de entrenamiento con clases separadas

Lote: `data/annotations/train-separate-v2-01/`. Contiene 8 imágenes de 8 grupos originales de entrenamiento, 43 entidades propuestas y 15 relaciones pendientes. No se incorporó ninguna imagen de validación/test ni se cambió el dataset v1.

La selección reproducible prioriza dos imágenes por criterio: tamaño pequeño de cabeza, superposición de cajas de contexto, mezcla de etiquetas originales de casco y ejemplos de referencia seleccionados por hash. Los criterios geométricos no prueban oclusión ni corrección de etiquetas. La revisión visual de la vista conjunta mostró escenas con múltiples ocupantes y objetos distantes; no se cambiaron etiquetas por juicio del asistente. No se usaron modelos preentrenados ni modelos para generar las nuevas cajas.

## Revisión humana necesaria

Abrir `review.html`, seleccionar cada entidad y completar sus cajas como en el piloto anterior. Moto: vehículo visible completo, sin extender la caja para incluir a la persona. Persona ocupante: cuerpo visible completo, incluida cabeza. Cabeza: cabeza/casco. Conservar rectángulos ajustados a extremos visibles, sin inventar partes ocluidas. Varias personas pueden vincularse a la misma moto.

Las clases de casco originales son propuestas pendientes. Todas las relaciones empiezan sin moto elegida: seleccionar explícitamente la moto correspondiente. No confirmar relaciones por proximidad automática.

Además de revisar cada entidad y vínculo, marcar la nueva casilla de cobertura **solo si están representadas todas las motos y personas ocupantes visibles**. En escenas con pasajeros, niños o vehículos distantes pueden faltar entidades en las etiquetas de origen. El editor actual no agrega entidades nuevas: dejar la cobertura desmarcada y comunicar la imagen y la entidad faltante para ampliar el lote. Rechazar una propuesta no elimina la necesidad de representar un objeto real. Los peatones ajenos a motos no forman parte de la clase `persona_ocupante`.

Guardar avances con «Guardar borrador JSON». Al terminar, introducir identificador y usar «Validar y exportar revisión». Copiar el archivo como:

`data/annotations/train-separate-v2-01/train-separate-reviewed.json`

## Validación y preparación posterior

```powershell
.venv/Scripts/python.exe -m training.validate_annotation_batch --canonical data/annotations/train-separate-v2-01/draft.json --reviewed data/annotations/train-separate-v2-01/train-separate-reviewed.json
.venv/Scripts/python.exe -m training.prepare_separate_labels --reviewed data/annotations/train-separate-v2-01/train-separate-reviewed.json --output data/prepared/helmet-separate-v2-staging-01
```

El exportador exige revisión humana válida, hashes y grupos intactos, partición train, cobertura completa y ausencia de casos inciertos. Excluye la imagen completa si hay ambigüedades, para no entrenar objetos omitidos como fondo. No sobrescribe directorios existentes ni modifica las etiquetas originales.

Clases preparadas: `0 motocicleta`, `1 persona_ocupante`, `2 cabeza_con_casco`, `3 cabeza_sin_casco`. Son etiquetas distintas de la antigua `moto_con_ocupante`; no deben mezclarse con ella. El manifiesto de preparación aún no se conecta al entrenamiento antiguo: habrá que adaptar el cargador y la salida del detector a cuatro clases más fondo.

Este lote es inicial, no un dataset suficiente para afirmar mejoras. Antes de otro entrenamiento completo hacen falta más ejemplos revisados y validación independiente con la misma taxonomía. El próximo modelo se inicializará aleatoriamente; no se reanudarán pesos del baseline ni se cargarán pesos externos. No se inició entrenamiento mientras faltan etiquetas humanas.

Verificaciones realizadas: borrador estructuralmente válido, hashes de originales correctos, 22 pruebas unitarias aprobadas y sintaxis JavaScript del editor comprobada. La vista conjunta se inspeccionó; el editor nuevo no se probó visualmente en navegador.
