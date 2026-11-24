# Ampliación con el lote 04

Se incorporan 10 imágenes: las 8 elegibles inicialmente y Image_00709/Image_00456, cuya cobertura completa confirmó el usuario en el chat. La exportación original y el cambio de declaración se conservan en el lote. Image_02169/Image_00396 permanecen excluidas.

Dataset: 35 imágenes train, 205 cajas (57 motos, 74 personas, 31 cabezas con casco y 43 sin casco). Validación original idéntica: 5 imágenes y 9 cabezas. Sin solapamiento con el lote test por identificador, grupo o hash.

Entrenamiento nuevo desde inicialización aleatoria, sin cargar pesos preentrenados ni checkpoints anteriores. Se comprobó el mismo hash de inicialización que el control. Se mantienen 150 épocas, batch 2, semilla 20261124, AdamW lr 0.0003 constante, aumentos fotométricos, paso 8 y propuestas ×4.

El aumento de 25 a 35 imágenes cambia los pasos de optimización de 1950 a 2700. Esta comparación no separa el efecto de los nuevos datos del mayor número de actualizaciones. Selección por mAP global de validación, sin cambiar el criterio.

Mejor época nueva: 108; duración: 483.8 segundos.

| AP@0.50 | Anterior, 25 imágenes | Nuevo, 35 imágenes |
|---|---:|---:|
| motocicleta | 57.43 % | 63.54 % |
| persona_ocupante | 58.25 % | 56.35 % |
| cabeza_con_casco | 33.66 % | 100.00 % |
| cabeza_sin_casco | 50.50 % | 37.87 % |
| mAP@0.50 | 49.96 % | 64.44 % |

AP no es exactitud ni mAP50:95. Los diagnósticos de propuestas y pérdidas por componente se conservan en los JSON. Las 29 pruebas unitarias pasaron antes de entrenar.

Las cinco imágenes de validación se han consultado repetidamente: estos resultados son exploratorios. El test previamente evaluado no se volvió a usar en esta comparación. El conteo sigue deshabilitado.

Reproducción:

```powershell
.venv/Scripts/python.exe training/train.py --task detector --separate-classes --dataset data/prepared/helmet-separate-v2-expanded-02 --epochs 150 --batch-size 2 --workers 0 --seed 20261124 --constant-lr --head-probe-every 10 --photometric --feature-stride 8 --proposal-multiplier 4 --run detector-separate-v2-expanded-02
```

Usar un nombre de ejecución nuevo para repetir; el entrenador rechaza sobrescrituras.

## Diagnóstico del mejor checkpoint

En validación, las propuestas RPN cubren 6/9 cabezas a IoU ≥0,50 y las predicciones finales con clase correcta cubren 6/9; al exigir confianza ≥0,50 quedan 3/9. Son coberturas por referencia, no recall uno a uno. La clasificación usando cajas reales acierta 1/9 y solo es un diagnóstico. Ver `head-diagnostics/diagnosis.json`.

El mAP mejora 14,48 puntos porcentuales frente al control, pero el AP sin casco baja 12,63 puntos. El AP con casco del 100 % corresponde únicamente a tres cabezas. No se habilita conteo con estos resultados.
