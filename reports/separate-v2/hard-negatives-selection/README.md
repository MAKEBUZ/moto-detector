# Selección de negativos difíciles

Minado de 160 imágenes elegibles de siete grupos originales de entrenamiento, excluyendo hashes y grupos de todos los lotes previos. Dos checkpoints propios congelados: reserva44/20261125 last y detector-separate-v2-expanded-02 best. RPN4000; sin entrenamiento. Se obtuvieron 696 predicciones de cabeza con score ≥0.05 e IoU <0.1 con las cabezas de origen: son candidatos, no errores confirmados, porque las etiquetas pueden omitir objetos.

Se inspeccionaron visualmente 21 escenas, tres por grupo, y se seleccionaron seis escenas de seis grupos: dos asientos, dos prendas, dos espejos/piezas. El séptimo grupo no aportó un ejemplo tan claro en la lista inspeccionada. El lote contiene 21 entidades y siete relaciones; todas pendientes de revisión humana. No se añaden clases nuevas ni se incorporan negativos automáticamente.

El editor integra una región sospechosa por imagen, selectores de decisión/categoría y exportación de un único JSON. El validador comprueba inventario y geometría de sugerencias, decisiones pendientes y cobertura incompleta ante dudas/cabezas reales. Los hashes de archivos val* preparados permanecen idénticos; ninguna imagen de validación se usó en el minado.

Artefactos: candidates.json (inventario), protocol.json (reglas y hashes), mining.json (predicciones), visual-candidates.json y mined-1/2/3.jpg (galerías), selection.json (selección). Checkpoint SHA256 registrado por predicción; parámetros y buffers verificados sin cambios. Lote: data/annotations/train-hard-negatives-v2-01; instrucciones LEEME.md.

Scripts: training/mine_hard_negative_review.py; training/build_training_annotation_batch.py con --selection-file; training/add_hard_negative_review.py. El minado requiere candidates.json; el constructor vuelve a comprobar elegibilidad. No sobrescribir este lote. Validación estructural correcta; 40 pruebas pasan, incluidas cuatro para las regiones negativas.
