# Dataset ampliado para el siguiente entrenamiento desde cero

Ruta: `data/prepared/helmet-separate-v2-expanded-01/`.

El lote train-separate-v2-03 pasó validación estructural, hashes y referencias humanas. Se aceptaron 11 de 12 imágenes. Image_00677 queda excluida porque la cobertura completa no fue confirmada; no se cambió la decisión del revisor.

| Partición | Imágenes | Cajas | Vínculos |
|---|---:|---:|---:|
| Train | 25 | 123 | 45 |
| Val | 5 | 25 | 9 |

| Clase | Train | Val |
|---|---:|---:|
| motocicleta | 33 | 7 |
| persona_ocupante | 45 | 9 |
| cabeza_con_casco | 24 | 3 |
| cabeza_sin_casco | 21 | 6 |

Entrenamiento crece de 14 a 25 imágenes y de 25 a 45 cabezas. Se conservaron íntegramente las 14 filas anteriores. La validación mantiene exactamente los mismos manifiesto, COCO e imágenes: se comprobó identidad de bytes de manifiesto y COCO, y hashes de todos los archivos de ambas versiones. No hay IDs/hashes duplicados ni grupos compartidos entre train y val. La independencia absoluta entre escenas depende de la agrupación original.

La nueva carpeta contiene imágenes, manifiestos con vínculos, COCO, estadísticas/procedencia (`dataset.json`) y hashes (`dataset.lock.json`). El dataset anterior y sus checkpoints permanecen intactos. El reporte persistente es `reports/separate-v2/helmet-separate-v2-expanded-01-prepared.json`.

Verificaciones: cargador de cuatro clases ejecutado sobre las 30 imágenes, etiquetas 1–4, cajas válidas tras escalado, locks de ambas versiones correctos y 25 pruebas unitarias aprobadas.

El siguiente entrenamiento debe usar `--separate-classes --dataset data/prepared/helmet-separate-v2-expanded-01` y un nombre de ejecución nuevo. El entrenador existente inicializa pesos aleatoriamente y no reanuda checkpoints. No se inició otro entrenamiento durante esta ampliación, ni se modificaron pesos o umbrales. Aún es un dataset pequeño: su ampliación no garantiza resolver el fallo de cabezas; conviene medir cobertura de propuestas y pérdidas por componente en el siguiente experimento.
