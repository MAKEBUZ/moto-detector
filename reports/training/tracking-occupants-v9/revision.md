# Asociación de los tres cascos pendientes

Se repitió el video panorámico con el checkpoint propio `joint-occupant-negatives-11`, las mismas referencias de v8 y el umbral de IoU de 0,50. Un casco cuenta como recuperado cuando coincide con la referencia en el fotograma marcado, está vinculado a su ocupante y esa trayectoria termina confirmada «con casco».

**Se recuperó uno de los tres cascos pendientes.** El casco blanco del fotograma 150 quedó vinculado a la trayectoria 6 con una IoU de 0,528. Cada una de las siete marcas que v8 ya resolvía sigue vinculada al mismo motorista. [Los tres casos](tres-cascos.jpg) · [Resultados de las diez marcas](additional-frames.json)

| Comprobación en el mismo video | v8 | v9 |
|---|---:|---:|
| Cascos de control vinculados en el fotograma y confirmados al final | 7/10 | **8/10** |
| Cascos de la revisión original, en trayectorias distintas confirmadas | 9/9 | **9/9** |
| Regiones negativas con ocupante asociado en el fotograma | 3/8 | 3/8 |
| Regiones negativas con falso ocupante en el conteo final | 2/8 | 2/8 |
| Regiones negativas con ocupante clasificado «con casco» al final | 0/8 | 0/8 |
| Trayectorias de moto / de ocupante | 43 / 40 | 43 / 41 |
| Trayectorias con casco / sin casco / indeterminadas | 19 / 0 / 21 | 20 / 0 / 21 |

Las 40 trayectorias de ocupante de v8 conservaron su intervalo, sus observaciones, sus votos y su estado final. La única trayectoria añadida muestra al mismo motorista blanco en 140 fotogramas entre el 36 y el 182. Hubo diez detecciones directas de casco y torso en los fotogramas 26, 31, 36, 51, 61, 101, 111, 121, 126 y 146. La inspección visual de las 140 cajas permitió comprobar que se contó una sola vez. La ejecución completa procesó 2,87 fotogramas/s en este equipo. [Marcas originales y negativas](comparison.json) · [Las 140 observaciones](observaciones.jpg) · [Coordenadas por fotograma](candidates.json)

### Motorista del borde inferior en el fotograma 150

El casco blanco y los hombros siguen visibles mientras el cuerpo sale parcialmente por abajo. La vista de la escena completa pierde a ese ocupante. El seguimiento v9 explora la franja inferior en recortes solapados. El detector propone el torso y el especialista vuelve a buscar el casco en un recorte más amplio de los píxeles originales. La entrada al seguimiento exige tres confirmaciones en fotogramas distintos y continuidad visual del casco y del torso. Si otra persona ya explica la detección, la app descarta la propuesta adicional.

Una primera variante encontró el casco blanco pero perdió una marca previa en el fotograma 750. La búsqueda adicional ahora sigue los puntos del casco y del torso sin volver a segmentar el contorno usado por el seguimiento anterior. Otra variante contó dos veces al motorista blanco cuando venció el plazo entre detecciones directas. En v9, después de tres confirmaciones, el seguimiento puede mantener el mismo identificador hasta nueve intervalos de búsqueda. Pasados cuatro intervalos, vuelve a clasificar el casco en los píxeles actuales antes de mostrarlo. Las comprobaciones del movimiento y del aspecto del torso se aplican en cada fotograma.

### Conductor y pasajero en el fotograma 450

Se buscaron ambos cascos desde los lados de la caja compartida y se probó el detector de personas en tres recortes centrados en la propuesta de casco delantero. Las tres cajas de cuerpo obtuvieron una IoU de 0,69 a 0,85 respecto a la caja compartida y las tres todavía admitían el casco del pasajero. La caja de cabeza elegida en este sondeo obtuvo una IoU de 0,475 con la referencia del conductor. [Tres recortes del mismo grupo](dos-ocupantes.jpg)

Se probaron seis recortes adicionales tomando como referencia el casco ya asignado al pasajero. Se varió la altura del corte para ocultar parte de ese casco y se ensayaron dos márgenes laterales. Las seis cajas de persona seguían admitiendo al pasajero y la mejor propuesta de cabeza tuvo una IoU de 0,156 con la referencia del conductor. Solo el pasajero cuenta como vinculado. [Coordenadas y puntuaciones de ambos sondeos](sondeo-dos-ocupantes.json)

También se buscaron cajas adyacentes que permitieran seguir al conductor desde otro fotograma. Entre el 365 y el 520, v8 tuvo una caja para el grupo en 144 fotogramas. En ocho se solapó otra caja con IoU superior a 0,15. La inspección de esos ocho recortes mostró que la caja adicional se extendía hacia la moto de atrás o abarcaba a ambos ocupantes. Vincularla al conductor habría mezclado personas. [Las ocho apariciones con ambas cajas](cuerpos-adyacentes.jpg)

### Motorista central en el fotograma 1350

La trayectoria de v8 había empezado sobre el frontal de la moto, en una región negativa del 1297. Más adelante, con el motorista a la vista, también hay un error de clasificación. En 52 fotogramas con caja de ocupante se buscó la cabeza con el especialista y se volvió a localizar con el detector general en otra vista. Se exigieron puntuaciones de al menos 0,65, una IoU mutua de al menos 0,55 y probabilidad superior a 0,92 para la misma clase con el clasificador local en el recorte y en su reflejo.

Hubo seis concordancias. Dos estimaron «con casco» y cuatro estimaron «sin casco» sobre un casco negro visible. Las otras 46 observaciones se abstuvieron. Estas etiquetas son resultados del sondeo y se mantuvieron fuera del conteo de la app. El caso combina la confusión del casco oscuro con visor y la asociación previa al frontal de la moto. [Los seis recortes clasificados](clasificacion-frontal.jpg) · [Fotogramas y resultados](sondeo-frontal.json) · [La región negativa y el casco en la misma trayectoria](../tracking-occupants-v8/trayectoria-49.jpg)

### Cómo repetir la medición

Desde la raíz del proyecto, el siguiente comando decodifica los 1.500 fotogramas por el mismo flujo que usa **Cargar video**. Lee del informe de v8 la ruta del archivo original y compara con las anotaciones guardadas.

```powershell
.\.venv\Scripts\python.exe training\evaluate_rider_tracking.py --partial reports\training\tracking-occupants-v8\video.json
```

El archivo utilizado es `15740042_3840_2160_60fps.mp4`, con SHA-256 `903f0791b1409831ae1a9078a1273805d3c1978285508303d4e788c8da627d98`. Las diez marcas de control y las ocho regiones negativas permiten comparar cambios sobre escenas conocidas. Para estimar el rendimiento en otras grabaciones hace falta evaluarlas con sus propias anotaciones.
