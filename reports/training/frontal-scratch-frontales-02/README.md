# Entrenamiento con las revisiones de ocupantes y frontales

Se aceptaron las 8 escenas de ocupantes solapados, las 8 de visores y las 5 de prueba. Se comprobaron identidad, vínculos, coordenadas y huellas de las imágenes contra los borradores. El conjunto de entrenamiento reunió 73 imágenes anteriores y 16 nuevas. Se conservaron las 5 imágenes de validación para seleccionar el checkpoint por mAP50. Las 5 imágenes de prueba representan 5 grupos excluidos de este entrenamiento. En la partición histórica de EdgeVision aparecen 43 archivos de entrenamiento de esos mismos grupos. La independencia frente a todo el historial de v9 no está garantizada.

Se entrenó Faster R-CNN de cuatro clases con todos los pesos inicializados aleatoriamente, tamaño 640, paso espacial 8, multiplicador de propuestas 4, lote 4 y variaciones fotométricas. En 60 épocas hubo 1.380 actualizaciones y 5.340 exposiciones. La época 37 obtuvo el mayor mAP50 de validación (47,82 %). Después se ajustaron únicamente las dos filas de clasificación de cabezas, partiendo de ese checkpoint propio. Se utilizaron los ocho frontales revisados, sus reflejos y las cajas anotadas de las 89 imágenes. Se eligió el paso 10 de 60 por mAP50 de validación. Las regiones de frontal se usaron como ejemplos negativos de cabeza, conservando la etiqueta de motocicleta. [Resultados del entrenamiento](../frontal-scratch-01/result.json) · [Ajuste con negativos](calibration.json)

| Cinco fotos de prueba | v9 | Detector nuevo con frontales |
|---|---:|---:|
| mAP50 del detector | 84,41 % | 43,83 % |
| Ocupantes acertados a confianza 0,50 | 7/8 | 4/8 |
| Falsos positivos de ocupante a confianza 0,50 | 0 | 3 |
| Cabezas con casco acertadas a confianza 0,50 | 1/2 | 0/2 |
| Cabezas sin casco acertadas a confianza 0,50 | 6/6 | 1/6 |
| Vínculos cabeza-persona correctos en el flujo de imágenes | 6/8 | 1/8 |

Para evaluar un vínculo, el cuerpo y la cabeza deben coincidir con sus referencias con IoU de al menos 0,50 y la clase del casco debe ser correcta. Se emparejan personas distintas dentro de cada foto. La magnitud de la muestra limita la comparación. [Resultados por escena](reserved.json)

## Repetir las dos fases

Con el entorno del proyecto y el dataset local preparado, las siguientes órdenes ejecutan las dos fases. Los directorios de salida deben estar libres antes de iniciar una repetición. Las imágenes y las etiquetas de cada partición están identificadas en los [manifiestos y huellas](../frontal-scratch-01/dataset.lock.json).

```powershell
.\.venv\Scripts\python.exe training\train.py --run frontal-scratch-01 --dataset data/prepared/frontal-scratch-01 --task detector --separate-classes --epochs 60 --batch-size 4 --workers 0 --feature-stride 8 --proposal-multiplier 4 --photometric
.\.venv\Scripts\python.exe training\train_reviewed_fronts.py
```
