# Ampliación de entrenamiento y validación v2

Se prepararon dos lotes pendientes de revisión humana:

| Lote | Partición original | Imágenes | Entidades propuestas | Relaciones pendientes |
|---|---|---:|---:|---:|
| train-separate-v2-02 | train | 12 | 77 | 28 |
| val-separate-v2-01 | val | 8 | 47 | 16 |

Los dos lotes usan las clases `motocicleta`, `persona_ocupante`, `cabeza_con_casco` y `cabeza_sin_casco`. Las cajas de moto y persona están vacías; las cabezas y sus clases se copian de las anotaciones originales y deben revisarse. No se generaron etiquetas con modelos ni se modificó el dataset v1. Las seis imágenes aceptadas anteriormente siguen en `data/prepared/helmet-separate-v2-staging-01/`.

## Selección y separación

Entrenamiento: tres imágenes por criterio de tamaño de cabeza, superposición de contexto, mezcla de etiquetas de casco y selección de referencia por hash. Validación: selección determinista por hash, sin usar puntuaciones ni errores del modelo. Ambas selecciones se limitan a imágenes elegibles del detector con cabezas y motos anotadas, hasta cinco cabezas y cuatro contextos; no son muestras representativas de todas las escenas.

Se eligió una imagen por grupo y se excluyeron los grupos y hashes presentes en todos los borradores anteriores, incluido el piloto de validación. Se verificaron los hashes de las imágenes originales y la coincidencia con `image-decisions.jsonl`. Esta separación respeta los grupos disponibles; no certifica independencia entre escenas si la agrupación original es incompleta. Test no se utilizó.

## Revisar

Abrir el `review.html` de cada carpeta en `data/annotations/`. Dibujar motos y personas por separado, revisar casco, asignar cada cabeza a su persona y moto y confirmar estados. Varias personas pueden compartir moto. La caja de persona incluye su cabeza y partes visibles; la caja de moto no se extiende para incluir a la persona. La caja gris es contexto antiguo, no una etiqueta nueva.

Marcar cobertura completa solo si no falta ninguna moto, persona ocupante ni cabeza visible que deba anotarse. Si faltan entidades, dejar cobertura desmarcada y comunicar la imagen y el objeto faltante: el editor actual permite editar las entidades existentes, pero no agregar nuevas. No resolver una ausencia rechazando un objeto real. Los casos inciertos quedan excluidos al preparar el dataset.

Exportar con identificador del revisor mediante «Validar y exportar revisión»; no basta guardar borrador. Cada lote produce un JSON para todas sus imágenes:

- `data/annotations/train-separate-v2-02/train-separate-v2-02-reviewed.json`
- `data/annotations/val-separate-v2-01/val-separate-v2-01-reviewed.json`

## Preparación tras recibir las revisiones

```powershell
.venv/Scripts/python.exe -m training.prepare_separate_labels --canonical data/annotations/train-separate-v2-02/draft.json --reviewed data/annotations/train-separate-v2-02/train-separate-v2-02-reviewed.json --split train --output data/prepared/helmet-separate-v2-staging-02
.venv/Scripts/python.exe -m training.prepare_separate_labels --canonical data/annotations/val-separate-v2-01/draft.json --reviewed data/annotations/val-separate-v2-01/val-separate-v2-01-reviewed.json --split val --output data/prepared/helmet-separate-v2-validation-01
```

Estos comandos todavía no se ejecutaron: los archivos humanos nuevos no existen. El exportador valida la revisión, comprueba partición/grupo/hash originales y excluye escenas incompletas o ambiguas. Conserva `val` como `val`; no la transforma en entrenamiento. Nunca sobrescribe un directorio de salida existente.

## Estado del próximo entrenamiento

No se ha iniciado entrenamiento ni se han cargado pesos nuevos. Primero deben aceptarse las anotaciones, consolidarse los manifiestos sin duplicados y adaptarse cargador, clases y métricas del detector. El modelo nuevo requerirá cuatro clases más fondo y pesos inicializados aleatoriamente, sin reutilizar el checkpoint anterior. Este volumen inicial permitirá comprobar el funcionamiento, pero no demostrar robustez ni habilitar automáticamente conteo o video.

Verificaciones: auditoría reproducible (`python -m training.audit_separate_batches`), 22 pruebas unitarias aprobadas, sintaxis JavaScript y nombres de exportación de ambos editores comprobados. No se probó visualmente la interacción en navegador.
