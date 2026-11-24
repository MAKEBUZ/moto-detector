"""Evaluate all prespecified final checkpoints on unchanged validation only."""
import hashlib
import json
import shutil
from pathlib import Path
import torch
from data import SeparateHelmetDataset
from models import build_detector, forbid_external_weights
from metrics import detection_ap50
from evaluate_reserved import counts_at_threshold


@torch.inference_mode()
def main():
    out = Path('reports/training/fixed-updates-01')
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    protocol = read(out/'protocol.json')
    torch.set_num_threads(4); forbid_external_weights()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    records = []; configs = []; initial_by_seed = {}
    for job in protocol['jobs']:
        run = Path('runs')/job['run']; root = Path(job['dataset'])
        assert hashlib.sha256((root/'dataset.lock.json').read_bytes()).hexdigest()==protocol['dataset_lock_sha256'][str(root)]
        for n,digest in read(root/'dataset.lock.json').items():
            assert hashlib.sha256((root/n).read_bytes()).hexdigest()==digest
        checkpoint = run/protocol['checkpoint']
        ck = torch.load(checkpoint,map_location='cpu',weights_only=True)
        c = ck['config'];configs.append(c)
        assert c['pretrained'] is False and c['initialization']=='random'
        assert ck['optimizer_updates']==protocol['optimizer_updates']==1950
        assert c['seed']==job['seed'] and c['train_examples']==job['images']
        if job['seed'] in initial_by_seed: assert initial_by_seed[job['seed']]==c['initial_state_sha256']
        initial_by_seed[job['seed']]=c['initial_state_sha256']
        steps={int(state['step']) for state in ck['optimizer']['state'].values()}
        assert steps=={1950},steps
        ds = SeparateHelmetDataset(root,'val',augment=False)
        samples=[ds[i] for i in range(len(ds))];targets=[t for _,t in samples]
        model=build_detector(5,c['feature_stride'],c['proposal_multiplier'])
        model.load_state_dict(ck['model']);model.to(device).eval();preds=[]
        for start in range(0,len(samples),2):
            preds.extend([{k:v.cpu() for k,v in p.items()} for p in model([x.to(device) for x,_ in samples[start:start+2]])])
        metrics=detection_ap50(preds,targets,(1,2,3,4))
        assert abs(metrics['map50']-ck['validation']['map50'])<1e-6
        row={**job,'checkpoint':protocol['checkpoint'],'epoch':ck['epoch'],
             'optimizer_updates':ck['optimizer_updates'],'image_presentations':ck['image_presentations'],
             'initial_state_sha256':c['initial_state_sha256'],
             'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             'metrics':metrics,'counts50':counts_at_threshold(preds,targets,.5),
             'counts05':counts_at_threshold(preds,targets,.05)}
        records.append(row)
        dest=out/job['run'];dest.mkdir(exist_ok=False)
        for n in ('config.json','history.json','result.json','requirements.lock.txt'):
            shutil.copyfile(run/n,dest/n)
        shutil.copytree(run/'head-probes',dest/'head-probes')
        (dest/'predictions.json').write_text(json.dumps([{'image_id':r['image_id'],'prediction':{k:v.tolist() for k,v in p.items()},'target':{k:v.tolist() for k,v in t.items()}} for r,p,t in zip(ds.rows,preds,targets)],indent=2),encoding='utf-8')
        del model,ck
    fixed=['batch_size','workers','constant_lr','learning_rate','photometric_policy','feature_stride','proposal_multiplier','head_probe_every','classes','parameters','max_updates']
    for key in fixed: assert all(c[key]==configs[0][key] for c in configs),key
    assert len(set(initial_by_seed.values()))==2
    pairs=[]
    for seed in initial_by_seed:
        old,new=[r for r in records if r['seed']==seed]
        pairs.append({'seed':seed,'nohelmet_ap_delta_points':100*(new['metrics']['per_class'][3]['ap50']-old['metrics']['per_class'][3]['ap50']),
                      'nohelmet_tp50_delta':new['counts50'][3]['tp']-old['counts50'][3]['tp']})
    summary={'protocol_sha256':hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),'records':records,'paired_deltas':pairs,'fixed_settings_verified':fixed,'test_used':False,'checkpoint_selection':'last at update 1950'}
    (out/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Comparación a 1950 actualizaciones y dos semillas','',
           'Cuatro ejecuciones nuevas desde cero, sin pesos preentrenados ni reanudación. Diseño fijado en protocol.json antes de entrenar. Datasets de 35/44 imágenes; semillas 20261124/20261125. Se evalúa siempre last.pt del paso 1950, no best.pt ni la mejor época para sin casco.', '',
           'Se conservaron AdamW lr 0.0003 constante, batch 2, paso 8, propuestas ×4, aumentos fotométricos y diagnóstico cada diez épocas más el final. Validación original de cinco imágenes con seis cabezas sin casco y tres con casco; sin test. Se verificó el contador AdamW de todos los parámetros en 1950, hashes e inicialización idéntica dentro de cada pareja y distinta entre semillas.', '',
           '| Semilla | Train | AP moto | AP persona | AP con casco | AP sin casco | mAP | Sin casco TP/FP/FN a 0,5 |', '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in records:
        c=r['counts50'][3]
        lines.append(f"| {r['seed']} | {r['images']} | "+' | '.join(f"{v['ap50']*100:.2f} %" for v in r['metrics']['per_class'])+f" | {r['metrics']['map50']*100:.2f} % | {c['tp']}/{c['fp']}/{c['fn']} |")
    lines+=['','## Diferencias pareadas, 44 menos 35 imágenes','']
    for p in pairs:lines.append(f"- Semilla {p['seed']}: AP sin casco {p['nohelmet_ap_delta_points']:+.2f} puntos; aciertos a 0,5: {p['nohelmet_tp50_delta']:+d}.")
    positive = sum(p['nohelmet_ap_delta_points'] > 0 for p in pairs)
    lines += ['', '## Decisión', '',
              f'La mejora de AP sin casco aparece en {positive} de las {len(pairs)} semillas evaluadas. No se demuestra una mejora estable. No se selecciona una semilla favorable ni se promueve un checkpoint nuevo para conteo.', '',
              'En la semilla original, ambos tamaños detectan una cabeza a confianza 0,05; mejora su ordenamiento y con 44 imágenes ese acierto supera 0,5. Con la segunda semilla, ambos omiten las seis cabezas sin casco incluso a 0,05; el de 44 imágenes añade un falso positivo por encima de 0,5. El promedio no debe ocultar esta inconsistencia.', '',
              'Los diagnósticos finales de ambas clases de cabeza siguen mostrando una brecha train/val: con 44 imágenes la cobertura de clase correcta y puntuación ≥0,5 es 69/89 y 62/89 en train, frente a 1/9 y 1/9 en val según semilla. Es cobertura por referencia de inferencia individual, no recall uno a uno ni una métrica exclusiva de sin casco.', '',
              'Siguiente paso propuesto: analizar en los cuatro checkpoints las seis referencias sin casco, separando propuestas RPN, clasificación y fondo, y auditar el muestreo de regiones positivas por clase antes de otro cambio de entrenamiento. Una comparación posterior que además iguale minibatches completos deberá ejecutar de nuevo ambos controles; no alterar retroactivamente estos resultados. Se mantienen pesos y etiquetas originales y el conteo deshabilitado.']
    mean=sum(p['nohelmet_ap_delta_points'] for p in pairs)/len(pairs)
    lines+=['',f'Promedio descriptivo de la diferencia de AP sin casco: {mean:+.2f} puntos. Dos semillas no permiten afirmar significación estadística ni estabilidad general.', '',
            '## Exposición y límites', '',
            '| Semilla | Train | Época final (parcial) | Actualizaciones | Imágenes procesadas |', '|---|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['seed']} | {r['images']} | {r['epoch']} | {r['optimizer_updates']} | {r['image_presentations']} |")
    lines+=['', 'El dataset de 35 imágenes termina en la época 109 parcial y el de 44 en la 89 parcial. El loader de 35 tiene un minibatch de una imagen por época; el de 44 usa siempre dos. Igualar pasos no iguala exactamente exposición, composición de minibatches, frecuencia de evaluación ni recorridos del RNG. No se alteró el sampler para conservar el entrenamiento anterior. Los monitores preservan el RNG de PyTorch.', '',
            'Estas cifras no deben confundirse con los máximos de validación de los experimentos anteriores: aquí el criterio es paso fijo. La validación es pequeña y reutilizada; no equivale a prueba independiente. AP50 no es exactitud ni COCO mAP50:95. No se ajustaron umbrales ni se habilitó conteo.', '',
            'Verificación: 29 pruebas unitarias y una ejecución real de un solo paso comprobaron el corte a mitad de época. La inferencia congelada reprodujo el mAP de los cuatro checkpoints. Configuraciones, pérdidas, diagnósticos, predicciones y hashes están junto a este informe.', '',
            'Reproducción: `.venv/Scripts/python.exe training/run_fixed_updates.py`; usa carpetas nuevas para repetir, porque el script rechaza sobrescribir resultados. Evaluación: `.venv/Scripts/python.exe training/report_fixed_updates.py`.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'records':records,'paired_deltas':pairs},indent=2))


if __name__=='__main__': main()
