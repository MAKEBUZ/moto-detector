import json
import hashlib
import shutil
from pathlib import Path


def main():
    output = Path('reports/training')
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for task in ('classifier', 'detector'):
        run = Path('runs') / f'{task}-baseline'
        result = json.loads((run / 'result.json').read_text())
        config = json.loads((run / 'config.json').read_text())
        history = json.loads((run / 'history.json').read_text())
        best = history[result['best_epoch'] - 1]
        assert not result['pilot'] and not result['test_used'] and not config['pretrained']
        records[task] = {'result': result, 'config': config, 'best': best}
        target = output / task
        target.mkdir(exist_ok=True)
        for file in ('result.json', 'config.json', 'history.json', 'requirements.lock.txt'):
            shutil.copy2(run / file, target / file)
    c, d = records['classifier'], records['detector']
    cm, dm = c['best']['validation'], d['best']['validation']
    text = f'''# Primer entrenamiento desde cero

Se completaron dos entrenamientos locales con inicialización aleatoria, sin cargar pesos preentrenados. GPU: {c['config']['gpu']}. Entorno: Python 3.14, PyTorch {c['config']['torch']} y torchvision {c['config']['torchvision']}.

## Resultados de validación

| Modelo | Entrenamiento | Validación | Épocas | Mejor época | Métrica |
|---|---:|---:|---:|---:|---|
| Clasificador CNN propio | {c['config']['train_examples']} recortes | {c['config']['validation_examples']} recortes | {c['result']['epochs']} | {c['result']['best_epoch']} | Macro-F1: {cm['macro_f1']:.4f}; accuracy: {cm['accuracy']:.4f} |
| Faster R-CNN con backbone propio aleatorio | {d['config']['train_examples']} imágenes | {d['config']['validation_examples']} imágenes | {d['result']['epochs']} | {d['result']['best_epoch']} | mAP@0.50: {dm['map50']:.4f} |

Estas métricas son de validación interna y se utilizaron para seleccionar el checkpoint. El conjunto test no se utilizó. No son una medición independiente en tráfico real ni una garantía de seguridad.

### Clasificador

| Clase | Precisión | Recall | F1 |
|---|---:|---:|---:|
| Con casco | {cm['precision'][0]:.4f} | {cm['recall'][0]:.4f} | {cm['f1'][0]:.4f} |
| Sin casco | {cm['precision'][1]:.4f} | {cm['recall'][1]:.4f} | {cm['f1'][1]:.4f} |

Matriz de confusión: filas reales, columnas predichas; orden con casco / sin casco.

```text
{cm['confusion_matrix_true_rows'][0]}
{cm['confusion_matrix_true_rows'][1]}
```

### Detector

| Clase | AP@0.50 | Cajas reales en validación |
|---|---:|---:|
'''
    for name, metric in zip(('Moto con ocupante', 'Cabeza con casco', 'Cabeza sin casco'), dm['per_class']):
        text += f"| {name} | {metric['ap50']:.4f} | {metric['ground_truth_boxes']} |\n"
    text += f'''
AP interpolada en 101 puntos con IoU 0,50 y hasta 100 detecciones por imagen. No equivale a COCO mAP@0.50:0.95. Es un primer experimento corto, con red pequeña y resolución de 384–640 píxeles; las cabezas lejanas pueden resultar difíciles.

## Evidencia y artefactos

- `runs/classifier-baseline/best.pt` y `runs/detector-baseline/best.pt`: checkpoints propios, seleccionados por validación.
- `last.pt`: última época de cada ejecución.
- Configuración, curvas por época y resultados resumidos copiados en las carpetas `classifier/` y `detector/` de este informe.
- Configuraciones con `pretrained: false`, semilla, huella de estado inicial y huellas de código/dataset.
- Los estados finales tienen huellas diferentes de los iniciales. Ambas ejecuciones iniciaron sus redes de nuevo; no continuaron los checkpoints de las pruebas pequeñas.
- Clasificador: {c['config']['parameters']:,} parámetros. Detector: {d['config']['parameters']:,} parámetros.
- Duración de bucles de entrenamiento/validación: {c['result']['seconds']:.1f} s y {d['result']['seconds']:.1f} s respectivamente; no incluye instalación ni preparación inicial.

Las bibliotecas de redes neuronales son herramientas de implementación, no modelos preentrenados. Se utilizó la arquitectura Faster R-CNN, pero su backbone, propuestas y cabezas se inicializaron aleatoriamente.

## Alcance y próximos pasos

El clasificador recibe un recorte de cabeza; no localiza por sí mismo al motociclista. El detector tiene la clase moto con ocupante, no cajas independientes de cada pasajero. No se evalúa casco abrochado correctamente.

Antes de integrar cámara y conteo: revisar errores en validación, decidir arquitectura/resolución y duración del siguiente experimento, y medir la cadena detector-clasificador. El conjunto de prueba debe permanecer reservado hasta fijar el experimento. Todavía falta una evaluación con cámara/video externos.

Véase `training/README.md` para repetir el entrenamiento y ejecutar inferencias con pesos propios.
'''
    (output / 'README.md').write_text(text, encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    hashes = {str(Path('runs') / f'{task}-baseline' / 'best.pt'): hashlib.sha256((Path('runs') / f'{task}-baseline' / 'best.pt').read_bytes()).hexdigest()
              for task in ('classifier', 'detector')}
    (output / 'checkpoint-hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    if (output / 'detector-example.jpg').exists():
        with (output / 'README.md').open('a', encoding='utf-8') as report:
            report.write('\n## Comprobación de inferencia\n\nSe recargaron ambos checkpoints propios con `weights_only=True` y se ejecutó una predicción con cada uno. El ejemplo del detector utiliza la primera imagen de validación por nombre, con umbral 0,5, sin seleccionar una imagen según su resultado. Detecta la moto y también una cabeza al fondo: antes de contar infracciones es necesario asociar las cabezas con los ocupantes de la moto. Esta imagen no demuestra que la cadena completa esté resuelta.\n\n![Ejemplo del detector entrenado desde cero](detector-example.jpg)\n')
    print(json.dumps({'classifier_macro_f1': cm['macro_f1'], 'classifier_accuracy': cm['accuracy'], 'detector_map50': dm['map50'], 'test_used': False}))


if __name__ == '__main__':
    main()
