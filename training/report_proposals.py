"""Compare larger RPN budgets with unchanged stride-8 and stride-16 controls."""
import hashlib
import json
import shutil
from pathlib import Path


def main():
    names=['detector-separate-v2-photo-01','detector-separate-v2-stride8-01','detector-separate-v2-proposals4-01']
    runs=[Path('runs')/n for n in names]
    read=lambda p:json.loads(p.read_text())
    configs=[read(r/'config.json') for r in runs]
    results=[read(r/'result.json') for r in runs]
    histories=[read(r/'history.json') for r in runs]
    best=[h[r['best_epoch']-1] for h,r in zip(histories,results)]
    keys=['epochs','seed','batch_size','workers','dataset_lock_sha256','initial_state_sha256','learning_rate','constant_lr','head_probe_every','classes','photometric_policy','parameters','feature_stride']
    assert all(configs[1][k]==configs[2][k] for k in keys)
    run=runs[-1];out=Path('reports/training/separate-v2-proposals4-01');out.mkdir(parents=True,exist_ok=True)
    for name in ('config.json','history.json','result.json','requirements.lock.txt'):shutil.copyfile(run/name,out/name)
    shutil.copytree(run/'head-probes',out/'head-probes',dirs_exist_ok=True)
    summary={'runs':names,'best':best,'last':[h[-1] for h in histories],'verified_fixed_settings':keys,
             'checkpoint_sha256':{n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in ('best.pt','last.pt')},'test_used':False}
    (out/'comparison.json').write_text(json.dumps(summary,indent=2))
    lines=['# Paso 8 con cuatro veces más propuestas','',
           'Entrenamiento desde cero con pesos aleatorios. Se mantienen 25 imágenes train, 5 val, 150 épocas/1950 pasos, batch 2, semilla 20261124, AdamW lr constante 0.0003, paso 8 y política fotométrica. Test no se utilizó. No se cargaron checkpoints anteriores ni pesos externos.', '',
           'Único cambio configurado frente al paso 8 anterior: presupuestos de propuestas RPN multiplicados por cuatro, tanto en entrenamiento como en evaluación. Train: antes/después NMS de 1000/500 a 4000/2000. Evaluación: de 500/200 a 2000/800. Son máximos, no cantidades garantizadas. Se conservan umbrales de NMS, anclas, muestreo de regiones y máximo de 100 detecciones finales.', '',
           'Se modifican conjuntamente los cuatro límites; el experimento no identifica cuál aporta el efecto. La inicialización, configuración restante y lock de datos fueron comprobados contra el control de paso 8.', '',
           f"Mejor época nueva: {results[-1]['best_epoch']}. Duración: {results[-1]['seconds']:.1f} segundos. Criterio de selección sin cambios: mAP global de validación.", '',
           '| AP@0.50 | Paso 16 foto | Paso 8 foto | Paso 8 propuestas ×4 |','|---|---:|---:|---:|']
    for i,cls in enumerate(configs[-1]['classes']):
        lines.append('| '+cls+' | '+' | '.join(f"{b['validation']['per_class'][i]['ap50']*100:.2f} %" for b in best)+' |')
    lines+=['| Promedio | '+' | '.join(f"{b['validation']['map50']*100:.2f} %" for b in best)+' |','',
            '## Diagnóstico al final', '', '| Medida | Paso 8 train /45 | Nuevo train /45 | Paso 8 val /9 | Nuevo val /9 |','|---|---:|---:|---:|---:|']
    for key in ('rpn_covered_iou50','correct_head_coverage_iou50_score50','oracle_correct'):
        lines.append('| '+key+' | '+' | '.join(str(histories[i][-1]['head_monitor'][s][key]) for i,s in [(1,'train'),(2,'train'),(1,'val'),(2,'val')])+' |')
    lines+=['','Cobertura por referencia a IoU ≥0.50; no recall uno a uno. Clasificación con cajas reales (`oracle_correct`) es diagnóstico, no inferencia normal. AP usa correspondencia uno a uno y salidas desde el umbral interno 0.05; no es exactitud ni COCO mAP50:95.', '',
            '## Verificación y alcance', '',
            '28 pruebas unitarias aprobadas, incluyendo límites de propuestas, parámetros iniciales idénticos y umbrales conservados. Configuración y pérdidas por componente están en los JSON; `head-probes/` conserva las mediciones periódicas. Checkpoints en runs/detector-separate-v2-proposals4-01/. Las ejecuciones anteriores permanecen intactas.', '',
            'Una semilla y cinco imágenes de validación consultadas repetidamente no permiten confirmar generalización. No se cambiaron etiquetas, no se usó test y el conteo sigue deshabilitado.', '',
            'Reproducción: training/train.py con --task detector --separate-classes --dataset data/prepared/helmet-separate-v2-expanded-01 --epochs 150 --batch-size 2 --workers 0 --seed 20261124 --constant-lr --head-probe-every 10 --photometric --feature-stride 8 --proposal-multiplier 4 y un --run nuevo.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'best_epoch':results[-1]['best_epoch'],'validation':best[-1]['validation'],'last_monitor':histories[-1][-1]['head_monitor']},indent=2))


if __name__=='__main__':main()
