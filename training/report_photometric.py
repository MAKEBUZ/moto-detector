"""Compare fixed-budget photometric augmentation with its unchanged baseline."""
import hashlib
import json
import shutil
from pathlib import Path


def main():
    base=Path('runs/detector-separate-v2-long-01');run=Path('runs/detector-separate-v2-photo-01')
    out=Path('reports/training/separate-v2-photo-01');out.mkdir(parents=True,exist_ok=True)
    load=lambda p:json.loads(p.read_text())
    a,b=load(base/'config.json'),load(run/'config.json')
    fixed=['epochs','seed','batch_size','workers','dataset_lock_sha256','initial_state_sha256','learning_rate','constant_lr','head_probe_every','classes']
    assert all(a[k]==b[k] for k in fixed)
    old,new=load(base/'history.json'),load(run/'history.json')
    oldresult,result=load(base/'result.json'),load(run/'result.json')
    oldbest,newbest=old[oldresult['best_epoch']-1],new[result['best_epoch']-1]
    for name in ('config.json','result.json','history.json','requirements.lock.txt'):shutil.copyfile(run/name,out/name)
    shutil.copytree(run/'head-probes',out/'head-probes',dirs_exist_ok=True)
    summary={'fixed_configuration_verified':fixed,'baseline_best':oldbest,'photometric_best':newbest,
             'baseline_last':old[-1],'photometric_last':new[-1],
             'first_positive_head_ap_epoch':next((r['epoch'] for r in new if any(c['ap50']>0 for c in r['validation']['per_class'][2:])),None),
             'checkpoint_sha256':{name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ('best.pt','last.pt')},'test_used':False}
    (out/'comparison.json').write_text(json.dumps(summary,indent=2))
    lines=['# Experimento de brillo, contraste y desenfoque','',
           'Entrenamiento desde cero, sin cargar checkpoints ni pesos externos. 25 imágenes train y las mismas 5 val; 150 épocas, 1950 pasos, semilla 20261124, batch 2, AdamW lr constante 0.0003, misma arquitectura y reflejos horizontales. La huella inicial y el lock del dataset coinciden con el control de 150 épocas.', '',
           'Único cambio experimental: brillo uniforme 0.85–1.15 y contraste uniforme 0.85–1.15; probabilidad 0.25 de desenfoque gaussiano con radio uniforme 0.2–0.8 píxeles. Se aplica después de reducir a 640 px y antes de la transformación interna del detector, exclusivamente en train. El generador aleatorio fotométrico es independiente para conservar la secuencia de reflejos. Diagnósticos train y validación no llevan aumentos.', '',
           f"Selección sin cambios: mejor mAP global de validación. Control: época {oldresult['best_epoch']}; fotométrico: época {result['best_epoch']}. Duración del nuevo bucle: {result['seconds']:.1f} segundos.", '',
           '| AP@0.50 | Control | Fotométrico |','|---|---:|---:|']
    for name,x,y in zip(b['classes'],oldbest['validation']['per_class'],newbest['validation']['per_class']):
        lines.append(f"| {name} | {x['ap50']*100:.2f} % | {y['ap50']*100:.2f} % |")
    lines += [f"| Promedio | {oldbest['validation']['map50']*100:.2f} % | {newbest['validation']['map50']*100:.2f} % |", '',
              '## Diagnóstico al final, época 150', '',
              '| Medida | Control train /45 | Nuevo train /45 | Control val /9 | Nuevo val /9 |','|---|---:|---:|---:|---:|']
    for key in ('rpn_covered_iou50','correct_head_coverage_iou50_score50','oracle_correct'):
        lines.append('| '+key+' | '+' | '.join(str(h['head_monitor'][s][key]) for s,h in [('train',old[-1]),('train',new[-1]),('val',old[-1]),('val',new[-1])])+' |')
    lines += ['', 'Cobertura por referencia a IoU ≥0.50, no recall con correspondencia uno a uno. `oracle_correct` suministra cajas reales: diagnóstico, no inferencia normal. AP evalúa las salidas normales mediante correspondencia uno a uno. No es exactitud ni COCO mAP50:95.', '',
              '## Alcance', '',
              'Los tres cambios fotométricos se prueban como una política conjunta; este experimento no separa sus efectos individuales. Una sola semilla y cinco imágenes de validación consultadas repetidamente no permiten confirmar generalización. No se usó test, no se cambiaron etiquetas y no se habilitó conteo.', '',
              'Historial y pérdidas por componente: `history.json`; diagnósticos cada diez épocas: `head-probes/`; comparación y hashes: `comparison.json`. Pesos nuevos en `runs/detector-separate-v2-photo-01/`. El control permanece intacto.', '',
              'Verificación: 26 pruebas unitarias aprobadas, incluyendo conservación de cajas, etiquetas y RNG de reflejos, y desactivación fotométrica en validación.', '',
              'Reproducción: training/train.py con --task detector --separate-classes --dataset data/prepared/helmet-separate-v2-expanded-01 --epochs 150 --batch-size 2 --workers 0 --seed 20261124 --constant-lr --head-probe-every 10 --photometric y un --run nuevo.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'best_epoch':result['best_epoch'],'validation':newbest['validation'],'last_monitor':new[-1]['head_monitor']},indent=2))


if __name__=='__main__':main()
