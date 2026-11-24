# Dataset preparado: helmet-baseline-v1

Esta versión sirve para un primer entrenamiento desde cero. Separa dos tareas y conserva los originales intactos. No es todavía el dataset final de toda la aplicación.

## Tareas y clases

| Tarea | Fuentes utilizadas | Clases |
|---|---|---|
| Detector | EdgeVision v1 | 0 `moto_con_ocupante`, 1 `cabeza_con_casco`, 2 `cabeza_sin_casco` |
| Clasificador de recortes | EdgeVision v1 | 0 `con_casco`, 1 `sin_casco` |

`moto_con_ocupante` conserva el significado de `BikeWithRider`: no es una caja de moto aislada ni una detección independiente de cada ocupante. Para cumplir la detección y conteo independiente de conductor/pasajeros habrá que añadir anotaciones de ocupante. `con_casco` expresa presencia visible, no correa abrochada, certificación o uso correcto.

Motorcycle Helmet OD se reserva para revisión: la muestra original incluye mosaicos de varias fotos con cajas que pueden abarcar fondo, personas lejanas o regiones incompletas. Sus piezas pueden reaparecer transformadas en otros mosaicos sin que coincidan sus hashes ni el dHash de la imagen completa. No se usa en entrenamiento, validación ni prueba de esta versión. Para recuperarlo será necesario revisar cajas y reconstruir la procedencia de cada pieza, u obtener una versión sin mosaicos.

## Política de depuración

- No se mezclan cajas de cabeza con cajas de persona. Mapua, LDrago, Proper Helmet Usage y Helmet Violation se reservan para revisión/reanotación, incluyendo la distinción de uso incorrecto. Motorcycle Helmet OD también queda reservado por la revisión de mosaicos descrita arriba.
- Los originales con geometría inválida se apartan completos. Quitar solo la caja dejaría un objeto visible sin etiquetar y generaría un falso negativo. No se recortan coordenadas sin inspección.
- Los archivos sin anotaciones no se aceptan automáticamente como fondo.
- Las etiquetas de cabeza de clases contrarias que comparten más del 50 % del área de la menor caja se consideran sospechosas. La imagen sale del detector y los recortes afectados salen del clasificador. Es una regla conservadora: también puede apartar dos personas solapadas correctamente etiquetadas.
- Un recorte debe tener al menos 24 píxeles por lado en la caja original. Se conserva resolución nativa; no se aplican aumentos, escalado ni normalización antes del split.
- Los recortes se generan de las cajas de cabeza originales, sin la heurística del 40 % superior de una persona y sin margen añadido. Los límites enteros se redondean hacia fuera.
- Las repeticiones exactas de imagen se marcan en el inventario; no se borran del repositorio fuente. Las anotaciones repetidas exactas se eliminan de la copia preparada cuando correspondan.

Cada decisión por imagen está en `reports/preparation/image-decisions.jsonl`; las decisiones por anotación en `annotation-decisions.jsonl`. Los motivos pueden solaparse y no deben sumarse para obtener el total de imágenes excluidas. `review-queue.jsonl` conserva los casos pendientes.

## Agrupación y particiones

Objetivo aproximado: 70 % entrenamiento, 15 % validación y 15 % prueba. Se asignan grupos completos, equilibrando cantidades de imágenes, fuentes y clases, con semilla fija `helmet-baseline-v1-2026-11-23`.

Las relaciones se construyen antes de separar:

1. SHA-256 idéntico.
2. Identificador de video explícito en el nombre y derivados de exportación con el mismo nombre base.
3. Componentes candidatos de similitud dHash, conservados juntos por precaución. Esto no los convierte en duplicados confirmados ni ocasiona su eliminación.
4. Bloques de 25 nombres consecutivos de EdgeVision, unidos además por las relaciones anteriores. Esta es una protección heurística adicional, no una identificación de escenas reales.

El detector y el clasificador comparten la partición del original. Todos los recortes de una misma imagen permanecen juntos, incluso si la imagen solo aporta datos a una tarea.

No se conocen todos los videos/escenas originales. Por ello las particiones son adecuadas para un **baseline interno**, pero no certifican independencia completa por escena, identidad o cámara. La evaluación final del proyecto necesitará una fuente de video/cámara externa reservada.

## Estructura

```text
data/prepared/helmet-baseline-v1/
  detector/
    data.yaml
    images/{train,val,test}/
    labels/{train,val,test}/
  classifier/
    {train,val,test}/{con_casco,sin_casco}/
  detector-manifest.jsonl
  classifier-manifest.jsonl
  classifier-export.jsonl
  groups.json
  training-contract.json
```

Las etiquetas del detector tienen formato YOLO normalizado. El clasificador usa carpetas de clase con orden explícito en `training-contract.json`. `data.yaml` incluye la ruta absoluta local; se debe actualizar `path` si se mueve el dataset.

Las imágenes del detector son copias independientes. Las del clasificador son nuevos JPEG de calidad 95. Ningún archivo preparado es un enlace duro a un original.

## Reproducción

Requiere los originales y manifiestos validados de la fase anterior, Node.js y PowerShell 7 en Windows.

```powershell
node --test scripts/dataset-utils.test.mjs
node scripts/prepare-dataset.mjs
./scripts/export-classifier-crops.ps1
node scripts/verify-prepared-dataset.mjs
./scripts/preview-prepared-dataset.ps1
```

El preparador se niega a sobrescribir una versión existente. Para reproducir, utilizar una copia limpia del proyecto con los originales, o conservar la versión existente y adaptar la ruta/nombre de la nueva versión. No borrar la única copia de una versión publicada.

## Contrato del entrenamiento

- Inicialización aleatoria en todos los componentes; no cargar checkpoints, backbones ni pesos externos preentrenados.
- Entrenar únicamente con `train`; aumentos de datos y ajuste de pesos de clase solo a partir de ese conjunto.
- Usar `val` para selección del modelo y umbrales. Reservar `test` para la evaluación final del experimento.
- Aplicar el mismo preprocesamiento determinista en validación y prueba.
- Registrar configuración, semilla, arquitectura, versiones, curvas y checkpoint propio.
- Evaluar recall, precisión y mAP del detector; F1 y matriz de confusión por clase del clasificador; posteriormente evaluar la cadena detector-clasificador y el desempeño en fuentes externas.

`training-contract.json` documenta estas restricciones; todavía no existe un entrenador que las haga cumplir automáticamente. En esta fase no se cargó ni entrenó ningún modelo.
