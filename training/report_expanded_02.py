"""Compare batch-04 expansion with the fixed proposals-x4 baseline."""
import hashlib
import json
import math
import shutil
from pathlib import Path


def main():
    names = ['detector-separate-v2-proposals4-01', 'detector-separate-v2-expanded-02']
    runs = [Path('runs') / n for n in names]
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    configs = [read(r / 'config.json') for r in runs]
    results = [read(r / 'result.json') for r in runs]
    histories = [read(r / 'history.json') for r in runs]
    fixed = ['epochs', 'batch_size', 'workers', 'seed', 'constant_lr', 'learning_rate',
             'photometric_policy', 'feature_stride', 'proposal_multiplier', 'head_probe_every',
             'classes', 'parameters', 'initial_state_sha256', 'initialization', 'pretrained']
    for key in fixed:
        assert configs[0][key] == configs[1][key], key
    assert configs[1]['initialization'] == 'random' and configs[1]['pretrained'] is False
    for name in ('val-manifest.jsonl', 'val-coco.json'):
        assert (Path(configs[0]['dataset']) / name).read_bytes() == (Path(configs[1]['dataset']) / name).read_bytes()
    best = [h[r['best_epoch'] - 1] for h, r in zip(histories, results)]
    steps = [math.ceil(c['train_examples'] / c['batch_size']) * c['epochs'] for c in configs]
    out = Path('reports/training/separate-v2-expanded-02')
    out.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'history.json', 'result.json', 'requirements.lock.txt'):
        shutil.copyfile(runs[1] / name, out / name)
    shutil.copytree(runs[1] / 'head-probes', out / 'head-probes', dirs_exist_ok=True)
    summary = {'runs': names, 'best': best, 'last': [h[-1] for h in histories],
               'fixed_settings_verified': fixed, 'validation_unchanged': True,
               'optimizer_steps': steps, 'test_used_in_this_comparison': False,
               'checkpoint_sha256': {n: hashlib.sha256((runs[1] / n).read_bytes()).hexdigest() for n in ('best.pt', 'last.pt')}}
    (out / 'comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = ['# Ampliación con el lote 04', '',
             'Se incorporan 10 imágenes: las 8 elegibles inicialmente y Image_00709/Image_00456, cuya cobertura completa confirmó el usuario en el chat. La exportación original y el cambio de declaración se conservan en el lote. Image_02169/Image_00396 permanecen excluidas.', '',
             'Dataset: 35 imágenes train, 205 cajas (57 motos, 74 personas, 31 cabezas con casco y 43 sin casco). Validación original idéntica: 5 imágenes y 9 cabezas. Sin solapamiento con el lote test por identificador, grupo o hash.', '',
             'Entrenamiento nuevo desde inicialización aleatoria, sin cargar pesos preentrenados ni checkpoints anteriores. Se comprobó el mismo hash de inicialización que el control. Se mantienen 150 épocas, batch 2, semilla 20261124, AdamW lr 0.0003 constante, aumentos fotométricos, paso 8 y propuestas ×4.', '',
             f'El aumento de 25 a 35 imágenes cambia los pasos de optimización de {steps[0]} a {steps[1]}. Esta comparación no separa el efecto de los nuevos datos del mayor número de actualizaciones. Selección por mAP global de validación, sin cambiar el criterio.', '',
             f"Mejor época nueva: {results[1]['best_epoch']}; duración: {results[1]['seconds']:.1f} segundos.", '',
             '| AP@0.50 | Anterior, 25 imágenes | Nuevo, 35 imágenes |', '|---|---:|---:|']
    for i, cls in enumerate(configs[1]['classes']):
        lines.append('| ' + cls + ' | ' + ' | '.join(f"{b['validation']['per_class'][i]['ap50']*100:.2f} %" for b in best) + ' |')
    lines += ['| mAP@0.50 | ' + ' | '.join(f"{b['validation']['map50']*100:.2f} %" for b in best) + ' |', '',
              'AP no es exactitud ni mAP50:95. Los diagnósticos de propuestas y pérdidas por componente se conservan en los JSON. Las 29 pruebas unitarias pasaron antes de entrenar.', '',
              'Las cinco imágenes de validación se han consultado repetidamente: estos resultados son exploratorios. El test previamente evaluado no se volvió a usar en esta comparación. El conteo sigue deshabilitado.', '',
              'Reproducción:', '', '```powershell',
              '.venv/Scripts/python.exe training/train.py --task detector --separate-classes --dataset data/prepared/helmet-separate-v2-expanded-02 --epochs 150 --batch-size 2 --workers 0 --seed 20261124 --constant-lr --head-probe-every 10 --photometric --feature-stride 8 --proposal-multiplier 4 --run detector-separate-v2-expanded-02', '```', '',
              'Usar un nombre de ejecución nuevo para repetir; el entrenador rechaza sobrescrituras.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'best_epoch': results[1]['best_epoch'], 'validation': best[1]['validation'], 'steps': steps}, indent=2))


if __name__ == '__main__':
    main()
