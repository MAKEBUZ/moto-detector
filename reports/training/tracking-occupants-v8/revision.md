# Cascos pendientes en los fotogramas de control

Se utilizó el mismo archivo panorámico de 1.500 fotogramas y el checkpoint propio `joint-occupant-negatives-11`. Las cajas de referencia y el umbral de IoU de 0,50 son los mismos que en v7. Cada casco de un fotograma requiere una trayectoria distinta, confirmada como «con casco» al finalizar el video.

**Se recuperaron dos de los cinco cascos pendientes.** El fotograma 750 quedó vinculado a la trayectoria 25 con IoU 0,685. En el 1350, el casco de la izquierda quedó vinculado a la trayectoria 51 con IoU 0,638. El primer vínculo recupera una aparición de un ocupante ya contado con casco y el segundo cambia el estado final del ocupante. [Comparación visual de los cinco casos](cinco-controles.jpg)

| Comprobación | v7 | v8 |
|---|---:|---:|
| Cascos confirmados en los fotogramas de control | 5/10 | 7/10 |
| Cascos confirmados de la revisión original | 9/9 | 9/9 |
| Regiones negativas con ocupante asociado en el fotograma | 3/8 | 3/8 |
| Regiones negativas con falso ocupante en el conteo final | 2/8 | 2/8 |
| Regiones negativas que terminaron «con casco» | 0/8 | 0/8 |
| Trayectorias de moto / de ocupante | 43 / 40 | 43 / 40 |
| Con casco / sin casco / indeterminadas | 18 / 0 / 22 | 19 / 0 / 21 |

Se cotejaron las identidades de las nueve marcas originales y el resultado individual de las ocho regiones negativas. Las 42 cajas añadidas corresponden a siete trayectorias. La inspección visual de los 42 recortes mostró un casco visible en cada caja. Las 40 trayectorias confirmadas conservan sus identificadores, sus detecciones originales y sus votos del detector. Solo la trayectoria 51 cambia de estado final. Estos controles cubren el video utilizado para desarrollar el ajuste. [Las 42 cajas añadidas](asociaciones.jpg) · [Marcas originales y negativas](comparison.json) · [Fotogramas de control](additional-frames.json)

| Fotograma | Observación en v7 | Qué se comprobó |
|---:|---|---|
| 150 | El motorista de casco blanco aparece en primer plano con el cuerpo cortado por abajo. Las cajas con seguimiento pertenecen a otros ocupantes. | En recortes del borde inferior aparecen propuestas parciales de cuerpo. En las vistas probadas del detector general no coincidieron una caja de cuerpo y una de casco con puntuaciones de al menos 0,40 para este motorista. |
| 450 | El conductor y el pasajero se solapan. El detector asigna una sola caja de ocupante al grupo y la enlaza al casco del pasajero. | El especialista localiza el casco delantero en un recorte ampliado. Falta una segunda caja de ocupante fiable para poder contar a ambos por separado. |
| 750 | La trayectoria 25 pierde su caja de cuerpo después del fotograma 746. Todavía se distinguen el casco y el torso. | Se siguieron puntos de ambas regiones y se volvió a buscar el casco en los píxeles del fotograma 750. |
| 1350, centro | Hay una caja de ocupante en la trayectoria 49, con votos contradictorios sobre el casco. Ese mismo identificador coincide con una región negativa en el fotograma 1297. | En el 1297 la caja encierra el frontal de la moto. En el 1350 pueden obtenerse propuestas de casco en recortes centrados sobre la referencia, pero la coincidencia de los dos detectores con la ventana generada por la app es insuficiente. Reclasificar toda la trayectoria también afectaría al caso negativo. |
| 1350, izquierda | La trayectoria 51 pierde su caja de cuerpo después del fotograma 1342. El casco negro sigue visible. | El clasificador cambia de respuesta según cuánto visor incluye el recorte. Se comprobó el movimiento de casco y torso y se volvió a localizar la cabeza con dos ventanas de distinto tamaño. |

Las pruebas con recortes centrados en una referencia sirven para diagnosticar el motivo de una pérdida. La app elige las ventanas a partir de las detecciones y de su movimiento. [Cajas del detector y referencias revisadas](diagnostico.jpg) · [Secuencias con las cajas de ocupante](continuidad.jpg) · [Región negativa del 1297 y casco central del 1350](trayectoria-49.jpg)
