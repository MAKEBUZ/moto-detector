# Inspección de nueve cabezas de validación

Se inspeccionaron las nueve cajas revisadas, las cinco imágenes completas de validación y dos ejemplos de entrenamiento por cabeza. Las parejas se eligieron por clase y cercanía del tamaño en píxeles de entrada del modelo; no son vecinos semánticos ni una búsqueda con modelos preentrenados. El checkpoint usado es el mejor por mAP global, época 72. No se cambió ninguna etiqueta ni peso.

## Comparación visual

- [Cabezas 1–3 y ejemplos de entrenamiento](comparison-1.jpg)
- [Cabezas 4–6 y ejemplos de entrenamiento](comparison-2.jpg)
- [Cabezas 7–9 y ejemplos de entrenamiento](comparison-3.jpg)
- [Contexto completo de las cinco imágenes](validation-context.jpg)

En cada fila, validación está a la izquierda y entrenamiento en las otras dos columnas. Los recortes se amplían para inspección, por lo que su tamaño mostrado no representa la resolución que recibe el detector. `size` es el lado menor tras el escalado interno; `oracle=0` significa fondo al suministrar la caja correcta, no una clasificación normal de la imagen recortada. El clasificador recibe características de la imagen completa.

## Hallazgos medidos

| Medida, checkpoint de época 72 | Train | Val |
|---|---:|---:|
| Cabezas | 45 | 9 |
| Mediana de lado menor en entrada del modelo | 51.20 px | 45.47 px |
| Cabezas menores de 16 px | 15 | 1 |
| Cabezas menores de 32 px | 18 | 3 |
| Cubiertas por propuestas RPN a IoU ≥0.50 | 36 | 2 |
| Clasificación correcta suministrando caja real | 32 | 0 |

Validación no contiene únicamente cabezas más pequeñas que entrenamiento: seis de nueve superan 32 píxeles y aun así fallan. Los nueve casos se clasifican como fondo al suministrar sus cajas. Por tanto, aumentar resolución o cambiar anclas no resuelve por sí solo el fallo observado de clasificación. El diagnóstico de anclas de la época 150 registró una ancla con IoU ≥0.50 en ocho de nueve casos: hay geometría disponible para casi todas, aunque las propuestas aprendidas no la conservan necesariamente.

## Casos inspeccionados

| Imagen / entidad | Observación visual | Evidencia e interpretación limitada |
|---|---|---|
| Image_01474 / h0, sin casco | Perfil, imagen desenfocada, fondo verde. | Hay ejemplos train de tamaño parecido; la RPN no logra IoU 0.50 y la caja real se clasifica como fondo. |
| Image_01474 / h1, con casco | Casco visto desde atrás, con desenfoque. | La RPN sí localiza una caja compatible (IoU aproximadamente 0.76), pero el clasificador rechaza incluso la caja real. |
| Image_00734 / h0, con casco | Vista frontal, visor transparente y cara visible. | Cabeza relativamente grande (aproximadamente 70 px); la RPN cubre la cabeza (IoU aproximadamente 0.66), pero se clasifica como fondo. |
| Image_01087 / h0, sin casco | Perfil definido en una imagen amplia de tráfico. | Aproximadamente 23 px en el modelo; la resolución y el contexto pueden influir. No hay propuesta compatible. |
| Image_01087 / h1, con casco | Perfil con casco oscuro en escena amplia. | Aproximadamente 29 px; fallo de propuestas y clasificación. |
| Image_01087 / h3, sin casco | Cabeza distante, poco detalle y oclusión. | Aproximadamente 10 px; caso claro para investigar resolución. No afirmar una corrección de casco solo a partir del recorte. |
| Image_02284 / h0, sin casco | Cabeza frontal, iluminación parcial y escena urbana. | Aproximadamente 74 px; no es un fallo exclusivo de objetos pequeños. |
| Image_01756 / h0, sin casco | Cabello visto desde atrás, dos ocupantes próximos. | Aproximadamente 81 px; los ejemplos train cercanos por tamaño sí pueden reconocerse. |
| Image_01756 / h1, sin casco | Cabeza parcialmente tapada por la otra ocupante. | Aproximadamente 64 px; revisar consistencia de criterio de caja visible/ocluida antes de cambiar etiquetas. |

Se inspeccionaron también las imágenes completas Image_01306 y Image_00086 de entrenamiento para evitar juzgar recortes ambiguos. La gorra del ocupante de fondo de Image_01306 no debe confundirse automáticamente con casco. No se propone ninguna corrección automática de clase. Las observaciones visuales del asistente no sustituyen anotaciones humanas.

## Decisión para la siguiente mejora

**Priorizar diversidad y regularización del entrenamiento antes de otro aumento de épocas.** La diferencia entre clasificación de cajas reales train/val es fuerte, incluso a tamaños comparables; es compatible con memorizar apariencia/contexto. No prueba qué atributo visual concreto causa el fallo.

El primer experimento controlado propuesto conserva dataset, validación, arquitectura y presupuesto de entrenamiento y añade únicamente una política moderada de variación fotométrica en train (brillo, contraste y desenfoque). Evaluar AP de ambas clases de cabeza, AP global y clasificación con cajas reales; informar todos los resultados sin cambiar el criterio de checkpoint después de verlos. Esto probaría sensibilidad a apariencia, no demostraría suficiencia del dataset.

En paralelo, ampliar entrenamiento revisado con más vistas frontales/de espaldas, cascos con visor, perfiles, oclusión entre ocupantes y distintas condiciones de imagen. Seleccionar de la partición train, conservar grupos y no trasladar estas nueve cabezas a entrenamiento. No afirmar que falte una categoría visual en todo train basándose solo en las dos parejas de cada fila.

**Segundo experimento, por separado:** mejorar resolución de características (por ejemplo, comparar paso 8 con el paso 16 actual, con inicialización aleatoria) para las cabezas pequeñas. Mantenerlo separado del cambio fotométrico para poder interpretar el resultado. No reducir únicamente el umbral ni dar por resuelto el problema con nuevas anclas: dos cabezas ya tienen propuestas correctas y aun así se rechazan.

La validación actual tiene cinco escenas y solo tres cabezas con casco, y se ha consultado repetidamente. Un próximo conjunto de evaluación revisado e independiente será necesario para confirmar cualquier mejora sin selección excesiva sobre estas nueve cabezas. Test no se utilizó en esta inspección.

## Reproducibilidad y alcance

`python training/compare_head_domains.py` genera las comparaciones a partir de `best-head-diagnostics/diagnosis.json` y los manifiestos del dataset. `comparison.json` conserva el hash del checkpoint, los nueve casos y sus ejemplos emparejados. Se verificó que hay nueve casos y que todos los vecinos son train de la misma clase. Los resultados de la época 150 se distinguen de los de la 72; no se mezclan para mejorar las métricas reportadas.

Esta tarea produjo inspección y decisión de experimento. No se reentrenó, no se modificaron etiquetas, ni se habilitó asociación o conteo.
