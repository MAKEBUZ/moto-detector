# Revisión de cascos en el borde

Se reprodujeron los 1.500 fotogramas de `15740042_3840_2160_60fps.mp4` con el detector `joint-occupant-negatives-11`. El SHA-256 del video es `903f0791b1409831ae1a9078a1273805d3c1978285508303d4e788c8da627d98`.

| Fotograma revisado | Última observación directa usada | Trayectoria | IoU con el casco marcado |
|---:|---:|---:|---:|
| 680 | 653 | 27 | 0,678 |
| 837 | 777 | 31 | 0,564 |

En el 680 el cuerpo sale por abajo. En el 837 el casco rojo y negro está parcialmente fuera del borde izquierdo y hay otro motociclista más arriba. Las dos asociaciones superan el mínimo de IoU de 0,50 y conservan su identidad al terminar el video. [Cajas recuperadas y referencias manuales](vista-bordes.jpg)

La revisión visual abarcó las 135 observaciones añadidas en 130 fotogramas, correspondientes a 11 trayectorias existentes. En los recortes, las cajas verdes siguen cascos visibles. El número debajo de cada recorte indica fotograma y trayectoria. Los identificadores 37 y 39 ilustran una fragmentación ya presente en v6. Los votos, el estado final y el número de detecciones de cada trayectoria confirmada coinciden con v6. [Recortes 1 a 64](auditoria-1.jpg) · [Recortes 65 a 128](auditoria-2.jpg) · [Recortes 129 a 135](auditoria-3.jpg) · [Detalle de las trayectorias](detalle.jpg)

De las ocho regiones negativas revisadas, tres coinciden con una caja de ocupante en el fotograma y dos llegan al conteo final, igual que en v6. Ninguna de las ocho termina con estado «con casco». La comprobación adicional conserva 5 de 10 cascos en otros cinco fotogramas del mismo clip. Queda pendiente medir la generalización a un video independiente. [Resultados por marca](comparison.json) · [Control adicional](additional-frames.json)
