# Revisión visual de las trayectorias 34 y 38

Se revisó el video original `15740042_3840_2160_60fps.mp4`. La identificación de cada ocupante se contrastó con la reproducción anotada de v9. Las imágenes de esta lámina son recortes del original. [Seis recortes](recortes.jpg)

| Trayectoria | Fotogramas | Observación |
|---|---|---|
| 34 | 824 y 830 | Se ve el casco oscuro del mismo conductor. |
| 34 | 879 | El conductor está saliendo por el borde izquierdo. Se aprecia ropa, sin una vista suficiente de la cabeza para verificar el casco. |
| 38 | 951, 962 y 968 | Se distingue el casco con visera del conductor en las tres vistas. |

Para resolver una trayectoria indeterminada, la regla de salida exige dos verificaciones directas separadas por al menos cinco fotogramas, con la caja de cabeza a más de dos píxeles del borde. Exige además al menos 12 votos de casco, cobertura en al menos el 25 % de las detecciones del ocupante y un 85 % de acuerdo entre los votos de cabeza. La evidencia adicional puede proceder de una doble localización o de tres comprobaciones por movimiento. El seguimiento sigue tomando sus decisiones con la regla anterior para evitar que una clasificación nueva se refuerce a sí misma.

Los controles de esta comparación corresponden al mismo video utilizado para ajustar la regla. [Comparación por marca revisada](comparacion.json)

En la primera prueba, la regla recuperó las dos trayectorias pero asignó un casco a una de las ocho regiones negativas. En el fotograma 1001, la trayectoria 38 ya cubre el asiento y el baúl de la moto. Los recortes anteriores sí muestran al conductor con casco. La regla final mantiene la abstención si las relocalizaciones contrarias abarcan cinco fotogramas o más. Para la trayectoria 38, el intervalo fue de 918 a 998. [Región negativa del fotograma 1001](negativo-1001.jpg) · [Primera prueba descartada](ensayo-sin-veto.json)
