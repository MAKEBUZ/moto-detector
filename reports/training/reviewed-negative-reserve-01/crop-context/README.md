# Tamaño y contexto de recortes con pesos congelados

Cajas reales, ventanas cuadradas centradas, borde replicado y dos resoluciones de entrada al backbone. El clasificador recibe la caja de cabeza en coordenadas del recorte. Cada columna binaria indica aciertos condicionados a una cabeza conocida. Fondo cuenta las veces que gana entre las cinco clases.

| Modelo | Partición | Ventana × lado mayor | Píxeles | Con casco | Sin casco | Fondo |
|---|---|---:|---:|---:|---:|---:|
| control | train | 1.0 | 64 | 21/39 | 52/57 | 95/96 |
| control | train | 1.0 | 128 | 31/39 | 48/57 | 76/96 |
| control | train | 1.5 | 64 | 20/39 | 48/57 | 92/96 |
| control | train | 1.5 | 128 | 32/39 | 51/57 | 43/96 |
| control | train | 2.0 | 64 | 21/39 | 45/57 | 93/96 |
| control | train | 2.0 | 128 | 32/39 | 52/57 | 48/96 |
| control | val | 1.0 | 64 | 2/3 | 6/6 | 8/9 |
| control | val | 1.0 | 128 | 2/3 | 5/6 | 6/9 |
| control | val | 1.5 | 64 | 2/3 | 5/6 | 9/9 |
| control | val | 1.5 | 128 | 2/3 | 5/6 | 5/9 |
| control | val | 2.0 | 64 | 2/3 | 5/6 | 9/9 |
| control | val | 2.0 | 128 | 2/3 | 6/6 | 6/9 |
| variant | train | 1.0 | 64 | 13/39 | 56/57 | 95/96 |
| variant | train | 1.0 | 128 | 32/39 | 48/57 | 84/96 |
| variant | train | 1.5 | 64 | 14/39 | 52/57 | 92/96 |
| variant | train | 1.5 | 128 | 34/39 | 52/57 | 40/96 |
| variant | train | 2.0 | 64 | 18/39 | 47/57 | 93/96 |
| variant | train | 2.0 | 128 | 27/39 | 53/57 | 45/96 |
| variant | val | 1.0 | 64 | 1/3 | 5/6 | 8/9 |
| variant | val | 1.0 | 128 | 2/3 | 3/6 | 8/9 |
| variant | val | 1.5 | 64 | 2/3 | 6/6 | 9/9 |
| variant | val | 1.5 | 128 | 3/3 | 3/6 | 5/9 |
| variant | val | 2.0 | 64 | 1/3 | 5/6 | 9/9 |
| variant | val | 2.0 | 128 | 2/3 | 4/6 | 4/9 |
