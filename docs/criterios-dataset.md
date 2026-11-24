# Criterios de datos — detección de casco

Estado: propuesta inicial para preparar el entrenamiento; no se han modificado las etiquetas originales.

## Alcance

Detectar motocicletas y ocupantes, clasificar presencia de casco, procesar imágenes/video, contar eventos y consultar estadísticas. Todos los componentes aprendidos se inicializarán aleatoriamente. No se usarán pesos externos, backbones preentrenados ni modelos preentrenados para generar anotaciones. El asistente inicial usará reglas y consultas a la base de datos.

## Contrato de anotación

| Entidad | Caja y regla |
|---|---|
| Motocicleta | Caja de la motocicleta visible. Una caja de moto más conductor no se considera equivalente sin revisión. |
| Ocupante | Caja de cada conductor o pasajero visible. No asumir que una moto tiene una sola persona. |
| Cabeza | Región de cabeza incluyendo el casco si está presente. No usar automáticamente el 40 % superior del cuerpo como etiqueta de cabeza. |
| Con casco | Casco sobre la cabeza. No implica que la correa esté abrochada. |
| Sin casco | Cabeza visible sin casco; una gorra o pañuelo no es un casco. Un casco en la mano no indica por sí solo el estado de la cabeza. |
| No determinable | Oclusión, resolución o evidencia insuficiente. Mantener como estado de revisión/salida; no convertir en negativo. |

Guardar identificadores de imagen original, fuente y versión, escena/video, ocupante y asociaciones con moto/cabeza. Conservar la etiqueta original y la decisión de revisión. No inferir asociaciones ambiguas.

La clase futura «uso incorrecto» requiere reglas observables, ejemplos específicos y acuerdo entre anotadores. No se inferirá de la clase agregada actual, que mezcla situaciones distintas.

## Particiones

- Agrupar originales, recortes y aumentos derivados antes de dividir.
- Separar por video/escena cuando se conozca; nunca repartir fotogramas vecinos entre entrenamiento y prueba.
- Detectar duplicados exactos y similares dentro de cada fuente y entre fuentes.
- Fijar un manifiesto de train/val/test; 70/15/15 es orientativo, no obligatorio si rompe grupos.
- Aplicar aumentos y sobremuestreo únicamente al entrenamiento.
- Reservar prueba para la evaluación final; usar validación para umbrales y selección de modelos.

## Condiciones para comenzar entrenamiento

1. Versiones, procedencia y licencias verificadas por fuente.
2. Archivos legibles y correspondencia imagen-etiqueta comprobada.
3. Cajas válidas y semántica uniforme para cada tarea.
4. Anotaciones completas de las clases de detección en las imágenes seleccionadas, o exclusión documentada.
5. Duplicados entre particiones resueltos; agrupación por escena documentada.
6. Muestra visual revisada por fuente, clase y partición.
7. Registro de clases minoritarias y condiciones poco representadas.
8. Configuración que demuestre inicialización aleatoria, sin descargas de pesos.

## Decisiones pendientes

Fecha de entrega, integrantes, GPU disponible, segundo dataset y si la rúbrica exige arquitectura propia además de entrenamiento desde cero. Estos datos no impiden la auditoría inicial.
