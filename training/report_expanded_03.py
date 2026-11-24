"""Report the 44-image random-initialization experiment against 35 images."""
import hashlib
import json
import shutil
from pathlib import Path


def main():
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    names = ['detector-separate-v2-expanded-02', 'detector-separate-v2-expanded-03']
    runs = [Path('runs') / n for n in names]
    configs = [read(p/'config.json') for p in runs]
    results = [read(p/'result.json') for p in runs]
    histories = [read(p/'history.json') for p in runs]
    fixed = ['epochs','batch_size','workers','seed','constant_lr','learning_rate',
             'photometric_policy','feature_stride','proposal_multiplier','head_probe_every',
             'classes','parameters','initial_state_sha256','initialization','pretrained']
    for key in fixed:
        assert configs[0][key] == configs[1][key], key
    assert configs[1]['initialization'] == 'random' and configs[1]['pretrained'] is False
    for n in ('val-manifest.jsonl','val-coco.json'):
        assert (Path(configs[0]['dataset'])/n).read_bytes() == (Path(configs[1]['dataset'])/n).read_bytes()
    out = Path('reports/training/separate-v2-expanded-03')
    out.mkdir(parents=True, exist_ok=True)
    review = read(out/'nohelmet-review/comparison.json')
    best = [h[r['best_epoch']-1] for h,r in zip(histories,results)]
    for name in ('config.json','history.json','result.json','requirements.lock.txt'):
        shutil.copyfile(runs[1]/name,out/name)
    shutil.copytree(runs[1]/'head-probes',out/'head-probes',dirs_exist_ok=True)
    summary = {'runs':names,'best':best,'last':[h[-1] for h in histories],
               'fixed_settings_verified':fixed,'validation_byte_identical':True,
               'optimizer_steps':[2700,3300],'test_used':False,
               'nohelmet_counts':{tag:review[tag]['counts'] for tag in ('old','new')},
               'checkpoint_sha256':{n:hashlib.sha256((runs[1]/n).read_bytes()).hexdigest() for n in ('best.pt','last.pt')}}
    (out/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines = ['# Entrenamiento con 44 imágenes', '',
             'Se incorporan las nueve imágenes completas del lote 05. Train: 44 imágenes, 246 cajas, 36 cabezas con casco y 53 sin casco. Val: las mismas cinco imágenes y nueve cabezas (tres con casco, seis sin casco).', '',
             'Ejecución nueva, sin cargar pesos externos ni checkpoints anteriores. Inicialización aleatoria con el mismo hash inicial que el control. Se conservan semilla 20261124, 150 épocas, batch 2, AdamW lr 0.0003 constante, paso 8, propuestas ×4, aumentos fotométricos y diagnóstico cada diez épocas. Selección por mAP global, sin cambiar el criterio para favorecer sin casco.', '',
             'Cambian los datos y las actualizaciones totales: 2700 → 3300. No es una comparación a igual número de pasos ni permite atribuir causalmente el cambio solo a nuevos ejemplos. No se ejecutó el control a paso exacto 1950 sugerido como experimento separado.', '',
             f"Mejor época nueva: {results[1]['best_epoch']}; duración: {results[1]['seconds']:.1f} segundos.", '',
             '| AP@0,50 | 35 imágenes | 44 imágenes |', '|---|---:|---:|']
    for i,cls in enumerate(configs[1]['classes']):
        lines.append('| '+cls+' | '+' | '.join(f"{b['validation']['per_class'][i]['ap50']*100:.2f} %" for b in best)+' |')
    lines += ['| mAP@0,50 | '+' | '.join(f"{b['validation']['map50']*100:.2f} %" for b in best)+' |', '',
              '## Sin casco: correspondencia uno a uno', '',
              'Inferencia congelada en batch 2 con la misma agrupación/padding que la evaluación del entrenador. Se reprodujo el mAP guardado de ambos checkpoints. IoU ≥0,5; seis cabezas reales.', '',
              '| Confianza | Antes TP / FP / FN | Ahora TP / FP / FN |','|---|---:|---:|']
    for threshold in ('0.05','0.5'):
        counts=[review[tag]['counts'][threshold] for tag in ('old','new')]
        lines.append('| '+threshold+' | '+' | '.join(' / '.join(str(c[k]) for k in ('tp','fp','fn')) for c in counts)+' |')
    lines += ['', '![Seis cabezas sin casco](nohelmet-review/comparison.jpg)', '',
              'Verde: referencia; rojo: predicción sin casco de mayor IoU; azul: con casco. Solo se dibujan predicciones con IoU >0,1 respecto de esa cabeza. Las puntuaciones del gráfico corresponden a la caja de mayor IoU, no necesariamente a la de máxima confianza.', '',
              'AP no es exactitud ni COCO mAP50:95. Cinco imágenes consultadas repetidamente y una sola semilla no confirman generalización. Test no se utilizó. No se ajustaron umbrales, no se modificaron etiquetas y el conteo permanece deshabilitado.', '',
              'Los JSON conservan pérdidas por componente, propuestas periódicas, predicciones y ranking sin casco. Las 29 pruebas unitarias pasaron. Checkpoints propios en runs/detector-separate-v2-expanded-03/.', '',
              '## Reproducción', '', '```powershell',
              '.venv/Scripts/python.exe training/train.py --task detector --separate-classes --dataset data/prepared/helmet-separate-v2-expanded-03 --epochs 150 --batch-size 2 --workers 0 --seed 20261124 --constant-lr --head-probe-every 10 --photometric --feature-stride 8 --proposal-multiplier 4 --run detector-separate-v2-expanded-03',
              '```', '', 'Usar un nombre nuevo para repetir; el entrenador rechaza sobrescribir una ejecución.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'best_epoch':results[1]['best_epoch'],'validation':best[1]['validation'],'nohelmet_counts':summary['nohelmet_counts']},indent=2))


if __name__ == '__main__':
    main()
