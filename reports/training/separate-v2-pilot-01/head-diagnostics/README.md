# Por qué el piloto obtiene AP cero en cabezas

Se inspeccionó el checkpoint seleccionado (época 18) sin actualizar pesos. Se usaron las 14 imágenes de entrenamiento sin aumentos y las 5 de validación. No se utilizó test. Las clases internas son fondo=0, moto=1, persona=2, cabeza con casco=3 y cabeza sin casco=4.

| Comprobación sobre cabezas | Train | Val |
|---|---:|---:|
| Cabezas revisadas | 25 | 9 |
| Con predicción final de cabeza a IoU ≥ 0.50, sin exigir casco correcto | 0 | 0 |
| Con predicción final de la clase correcta a IoU ≥ 0.50 | 0 | 0 |
| Cubiertas por alguna propuesta RPN retenida a IoU ≥ 0.50 | 0 | 0 |
| Clase correcta al introducir la caja real en el clasificador de regiones | 0 | 0 |
| Lado menor de cabeza <16 píxeles en la entrada interna | 7 | 1 |
| Lado menor de cabeza <32 píxeles en la entrada interna | 9 | 3 |

Los conteos de tamaño son acumulativos, no categorías para sumar. La mediana del lado menor es 50.74 píxeles en train y 45.47 en val. Por tanto, no todos los fallos son objetos pequeños.

## Interpretación

Hay dos obstáculos observados. Primero, las propuestas de regiones retenidas ya fallan al localizar cabezas: su mejor IoU en los 34 casos es 0.182. Segundo, incluso suministrando las cajas reales, el clasificador de regiones favorece fondo en 32 cabezas y persona en 2; nunca una clase de cabeza. Esta prueba con cajas conocidas es diagnóstica, no una métrica de detección utilizable en producción.

Las salidas finales se inspeccionaron con el umbral interno original del detector (0.05), no solo con el umbral de visualización 0.5. Por eso bajar únicamente el umbral de visualización no resuelve este resultado. No se modificó ningún umbral.

El diagnóstico demuestra que el fallo está presente también en entrenamiento. No permite atribuirlo únicamente a sobreajuste, tamaño de objetos, anclas o número de épocas. El conjunto tiene solo 25 cabezas de entrenamiento, y el backbone usa un mapa de características de paso 16; son factores a investigar mediante experimentos controlados, no causas ya demostradas. La cobertura RPN se midió después de filtrar propuestas y no distingue por sí sola problemas de anclas, puntuación o regresión.

## Próximas acciones

1. Revisar el nuevo lote `train-separate-v2-03`: 12 imágenes de 12 grupos nuevos, 65 entidades y 25 relaciones. Las clases originales de cabeza incluyen 9 con casco y 16 sin casco, todavía sin confirmar. La selección prioriza cabezas grandes con/sin casco, etiquetas mixtas y cabezas pequeñas. No se agregan automáticamente al entrenamiento.
2. Después de aceptar anotaciones completas, repetir un experimento desde cero con más ejemplos. Registrar pérdidas separadas de RPN y clasificador y cobertura de propuestas durante entrenamiento para detectar dónde deja de aprender.
3. Comprobar capacidad de aprender una muestra pequeña de entrenamiento y comparar posteriormente cambios de resolución, propuestas/anclas o características de mayor resolución, uno por experimento. Mantener validación reservada para evaluación, sin convertirla en entrenamiento.

## Nuevo lote para revisión

Abrir `data/annotations/train-separate-v2-03/review.html`. Completar moto y persona, revisar cabeza/casco y seleccionar los vínculos. Confirmar cobertura solo si no falta ninguna entidad. Si faltan personas o motos, dejar cobertura desmarcada y comunicar las faltantes; el editor aún no permite agregar entidades.

Guardar mediante «Validar y exportar revisión» con identificador. Colocar el resultado en `data/annotations/train-separate-v2-03/train-separate-v2-03-reviewed.json`. Un solo JSON contiene las 12 imágenes. El lote tiene hashes originales verificados, sin grupos ni hashes repetidos respecto a borradores anteriores, y conserva la partición train. Esta comprobación no certifica independencia absoluta de escenas.

## Reproducibilidad

`python training/diagnose_four_class_heads.py` genera `diagnosis.json`, con detalles de las 34 cabezas, probabilidades de la prueba con cajas reales y hash del checkpoint. La inicialización del modelo de diagnóstico carga exclusivamente el checkpoint propio entrenado desde cero. No se usan pesos externos, no se entrena y no se crean etiquetas mediante modelos.

El editor nuevo pasó comprobación de sintaxis JavaScript; el borrador pasó validación estructural y hashes. Pasaron 25 pruebas del proyecto. No se ha confirmado ninguna anotación en nombre del usuario ni iniciado otro entrenamiento.
