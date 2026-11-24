# moto-detector

Proyecto académico de detección de motocicletas y ocupantes, clasificación de casco, conteo y consulta de estadísticas. Todos los modelos se han entrenado desde cero, sin pesos preentrenados.

## Estado

Actualizado al **24 de noviembre de 2026**. Se completaron la depuración de datos, revisiones humanas de cajas y relaciones, entrenamientos desde cero y experimentos de generalización en la RTX 3060. La referencia actual para el detector de cuatro clases es **paso 16 con variaciones de brillo, contraste y desenfoque**, con mAP@0,50 de validación de **43,95 %**. El conteo, video y asistente de estadísticas siguen pendientes; test permanece reservado.

## Resultados acumulados

### Modelos iniciales

| Modelo | Datos de entrenamiento / validación | Resultado de validación |
|---|---|---|
| Clasificador CNN de recortes de cabeza | 2889 / 628 recortes | Macro-F1: 91,00 %; exactitud: 91,72 % |
| Detector inicial de tres clases | 1583 / 347 imágenes | mAP@0,50: 69,12 % |

El detector inicial reconoce **moto con ocupante** como una sola clase, además de cabeza con casco y sin casco. Sus resultados **no son directamente comparables** con los del detector nuevo: cambian las clases, las anotaciones y la validación. El clasificador requiere un recorte de cabeza; no detecta personas por sí solo. [Informe inicial](reports/training/README.md).

### Detector de cuatro clases

Clases: **motocicleta, persona ocupante, cabeza con casco y cabeza sin casco**. El dataset ampliado tiene 25 imágenes de entrenamiento, 123 cajas y 45 cabezas; la validación contiene 5 imágenes, 25 cajas y 9 cabezas (3 con casco y 6 sin casco). Se conservaron las particiones originales y se excluyeron imágenes con cobertura incompleta o incertidumbre.

Las cifras siguientes corresponden al checkpoint seleccionado por **mAP global de validación**, no al máximo independiente de cada clase.

| Experimento | Train | Épocas | Mejor época | Moto AP | Persona AP | Con casco AP | Sin casco AP | mAP@0,50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Piloto de cuatro clases | 14 | 20 | 18 | 42,57 % | 35,30 % | 0 % | 0 % | 19,47 % |
| Dataset ampliado | 25 | 20 | 4 | 42,57 % | 44,55 % | 0 % | 0 % | 21,78 % |
| Mayor exposición, paso 16 | 25 | 150 | 72 | 68,98 % | 59,98 % | 0 % | 0 % | 32,24 % |
| **Paso 16 + variaciones fotométricas** | **25** | **150** | **105** | **69,70 %** | **55,60 %** | **33,66 %** | **16,83 %** | **43,95 %** |
| Paso 8 + variaciones fotométricas | 25 | 150 | 96 | 57,43 % | 45,64 % | 33,66 % | 16,83 % | 38,39 % |

Todos estos experimentos usan la misma validación de cinco imágenes. Los de 150 épocas mantienen tasa de aprendizaje constante de 0,0003; los de 20 épocas usaban planificación coseno. La política fotométrica aplica brillo y contraste entre 0,85 y 1,15 y desenfoque suave con probabilidad del 25 %, exclusivamente en entrenamiento. El experimento de paso 8 mantiene esa política y los demás parámetros; duplica la resolución espacial del mapa de características por eje.

**Conclusión actual:** las variaciones fotométricas dieron el mejor resultado observado. El paso 8 no mejoró el AP de cabezas y redujo el mAP global. La referencia de paso 16 conserva dos aciertos de cabeza con confianza inferior a 0,5; el paso 8 tiene un acierto por encima de 0,5. Ninguno justifica habilitar conteo de infracciones.

### Diagnóstico y asociación

- En el piloto humano de siete imágenes, el sistema propuso correctamente **5 de 11 vínculos confirmados cabeza→moto**. Los demás fallaron por localización o abstención ante clases contradictorias. La comparación usa el contexto antiguo moto+ocupante; no evalúa todavía una cadena completa cabeza→persona→moto.
- En una prueba de memorización con **4 imágenes y 7 cabezas**, el detector alcanzó AP@0,50 del **100 % en ambas clases** sobre la propia muestra entrenada, desde el paso 275. Esto confirma capacidad de aprender esos ejemplos, **no generalización**.
- Tras 150 épocas sin variaciones fotométricas, detectaba 39/45 cabezas de entrenamiento con clase correcta y confianza ≥0,5, frente a 0/9 en validación. Se observó una brecha compatible con sobreajuste.
- La inspección visual mostró fallos tanto en cabezas pequeñas como grandes. En algunos casos la propuesta localiza la cabeza, pero el clasificador la rechaza como fondo. El tamaño no explica por sí solo el problema.

### Límites y decisión

AP@0,50 mide detección a IoU 0,50; **no es exactitud ni COCO mAP@0,50:0,95**. Usa las predicciones del detector desde su umbral interno 0,05, por lo que no equivale al rendimiento con confianza mínima de 0,5. La validación es pequeña y se ha consultado repetidamente para elegir checkpoints; hace falta una evaluación independiente para confirmar cualquier mejora.

Se verificaron inicialización aleatoria, hashes de datos y checkpoints, métricas de cuatro clases y aislamiento de aumentos a entrenamiento. La última batería completó **27 pruebas**. Los pesos propios están en `runs/`, excluido de Git junto con los datasets descargados y preparados; los informes y anotaciones revisadas sí están versionados.

Se conserva paso 16 con política fotométrica como referencia. El siguiente análisis propuesto es estudiar las propuestas antes y después de top-k/NMS y confirmar resultados con mayor diversidad revisada. No se ha implementado la asociación de cuatro clases ni habilitado conteo, video, asistente de estadísticas o evaluación del casco correctamente abrochado.

## Informes de los experimentos

- [Piloto de cuatro clases](reports/training/separate-v2-pilot-01/README.md)
- [Entrenamiento con dataset ampliado](reports/training/separate-v2-expanded-01/README.md)
- [Prueba de memorización de cabezas](reports/training/head-overfit-01/README.md)
- [Entrenamiento prolongado](reports/training/separate-v2-long-01/README.md)
- [Inspección de cabezas de validación](reports/training/separate-v2-long-01/head-comparison/README.md)
- [Brillo, contraste y desenfoque](reports/training/separate-v2-photo-01/README.md)
- [Características con paso 8](reports/training/separate-v2-stride8-01/README.md)
- [Comparación de vínculos revisados](reports/relations-pilot/README.md)
- [Dataset ampliado de cuatro clases](docs/dataset-separado-v2-ampliado.md)

## Documentación de datos y preparación

- [Informe de auditoría](reports/auditoria-dataset.md)
- [Criterios de anotación y preparación](docs/criterios-dataset.md)
- [Resultados automáticos](reports/dataset-audit.json)
- [Recuperación y validación de originales](reports/recuperacion-originales.md)
- [Instrucciones para repetir las descargas](docs/recuperar-originales.md)
- [Fuentes y versiones fijadas](data/sources.lock.json)
- [Resultado de la depuración y particiones](reports/preparation/README.md)
- [Guía del dataset preparado](docs/dataset-preparado.md)
- [Primer entrenamiento y resultados](reports/training/README.md)
- [Entrenadores e inferencia desde cero](training/README.md)
- [Revisión de errores y asociación cabeza–moto](reports/error-review/README.md)

## Reproducir la auditoría

Requiere Git y Node.js, sin paquetes adicionales. Desde la raíz del proyecto:

```powershell
git clone --depth 1 https://github.com/Valery-Rosero/DataSetsIA.git data/sources/DataSetsIA
node scripts/audit-dataset.mjs
powershell -ExecutionPolicy Bypass -File scripts/preview-crops.ps1
```

El JSON registra el commit auditado. Para reproducir exactamente un informe, utilizar ese commit de la fuente. La muestra visual se genera en Windows con System.Drawing; no sustituye la revisión de todos los datos.

`data/sources/` está excluido de Git. Los scripts de auditoría no modifican las etiquetas originales. Las copias en `reports/upstream-reference/` se conservan como evidencia del código y los conteos publicados; no se deben ejecutar desde allí.

El repositorio fuente contiene etiquetas y recortes; las imágenes originales ya se recuperaron por separado en `data/sources/originals/`. No ejecutar su unificador sobre la única copia disponible: elimina su salida anterior antes de regenerarla.
