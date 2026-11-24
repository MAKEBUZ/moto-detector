# Primera evaluación reservada del detector

Las ocho imágenes revisadas pasaron validación y cobertura, sin exclusiones. Se evaluó exclusivamente el checkpoint fijado de época 136, paso 8, propuestas ×4, con su SHA256 verificado contra el protocolo. No se entrenó, no se eligió otro checkpoint y no se ajustaron umbrales. Esta parte de test ya fue evaluada; no debe describirse en adelante como prueba intacta.

| Clase | Cajas reales | AP@0.50 | TP | FP | FN | Precisión | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Motocicleta | 10 | 34.82 % | 3 | 1 | 7 | 75.00 % | 30.00 % |
| Persona ocupante | 11 | 25.84 % | 2 | 4 | 9 | 33.33 % | 18.18 % |
| Cabeza con casco | 4 | 38.12 % | 1 | 1 | 3 | 50.00 % | 25.00 % |
| Cabeza sin casco | 7 | 7.43 % | 0 | 1 | 7 | 0 % | 0 % |

**mAP@0.50: 26.55 %.** TP/FP/FN, precisión y recall se calculan con confianza ≥0.5 e IoU ≥0.5 mediante correspondencia uno a uno. AP utiliza las salidas desde el umbral interno del detector y no equivale a exactitud ni a COCO mAP50:95. Por eso puede haber AP positivo sin aciertos al umbral operativo 0.5.

La RPN cubre 7 de 11 cabezas a IoU ≥0.50, pero solo una cabeza tiene detección final de clase correcta por encima de confianza 0.5. Esta cobertura por referencia es diagnóstico, no recall de detecciones uno a uno.

## Interpretación

El mAP es inferior al 49.96 % de la validación utilizada para seleccionar el checkpoint. No es una regresión de entrenamiento: se evaluó el mismo modelo en imágenes distintas. La diferencia muestra que el resultado de desarrollo no se trasladó de manera suficiente a esta muestra reservada. La clase sin casco resulta especialmente débil y no justifica conteo de infracciones.

Ocho imágenes y once cabezas no representan todo el tráfico. La selección excluye escenas sin objetos anotados y no certifica independencia absoluta entre escenas si los grupos originales son incompletos. Se respetaron las particiones y el protocolo fijado; no hay garantías operativas derivadas de esta muestra.

Decisión: no habilitar conteo ni multas. Para continuar mejorando, trabajar con nuevos datos de entrenamiento/desarrollo revisados y mayor diversidad. No convertir este resultado en otra búsqueda de umbrales o checkpoints sobre las mismas imágenes. Una evaluación futura independiente requerirá un nuevo conjunto reservado.

## Evidencia

`results.json`: métricas, hashes de protocolo/checkpoint/revisión y código; `predictions.json`: todas las predicciones y referencias en píxeles de entrada redimensionada; `head-diagnostics.json`: diagnóstico de propuestas. Las etiquetas preparadas conservan split test en `data/prepared/helmet-separate-v2-test-01/`.

El evaluador `training/evaluate_reserved.py` rechaza sobrescribir este directorio. Pasaron 29 pruebas unitarias, incluyendo duplicados, clase incorrecta y umbral de confianza en las métricas. Los pesos anteriores y originales permanecen intactos.
