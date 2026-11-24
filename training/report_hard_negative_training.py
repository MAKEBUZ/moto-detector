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
    out = Path('reports/training/hard-negatives-fixed1950-01')
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    protocol = read(out/'protocol.json')
    torch.set_num_threads(4); forbid_external_weights()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    protocol['jobs']=[job for seed in (20261124,20261125) for job in protocol['controls_reused']+protocol['jobs'] if job['seed']==seed]
    records = []; configs = []; initial_by_seed = {}
    for job in protocol['jobs']:
        run = Path('runs')/job['run']; root = Path(job['dataset'])
        assert hashlib.sha256((root/'dataset.lock.json').read_bytes()).hexdigest()==protocol['dataset_lock_sha256'][str(root)]
        for n,digest in read(root/'dataset.lock.json').items():
            assert hashlib.sha256((root/n).read_bytes()).hexdigest()==digest
        checkpoint = run/protocol['checkpoint']
        ck = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if 'checkpoint_sha256' in job: assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==job['checkpoint_sha256']
        c = ck['config'];configs.append(c)
        assert c.get('rpn_object_reserve',False) is False
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
        assert ck['image_presentations']==3900
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
    fixed=['batch_size','workers','constant_lr','learning_rate','photometric_policy','feature_stride','proposal_multiplier','head_probe_every','classes','parameters','max_updates','torch','torchvision','device','num_classes_including_background']
    for key in fixed: assert all(c[key]==configs[0][key] for c in configs),key
    for name in ('data.py','models.py','metrics.py','overfit_heads.py'):
        source='training\\'+name
        assert all(c['source_sha256'][source]==configs[0]['source_sha256'][source] for c in configs),source
    assert len(set(initial_by_seed.values()))==2
    pairs=[]
    for seed in initial_by_seed:
        old,new=[r for r in records if r['seed']==seed]
        pairs.append({'seed':seed,'nohelmet_ap_delta_points':100*(new['metrics']['per_class'][3]['ap50']-old['metrics']['per_class'][3]['ap50']),
                      'nohelmet_tp50_delta':new['counts50'][3]['tp']-old['counts50'][3]['tp']})
    summary={'protocol_sha256':hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),'records':records,'paired_deltas':pairs,'fixed_settings_verified':fixed,'test_used':False,'checkpoint_selection':'last at update 1950'}
    (out/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Negativos difíciles: 44 frente a 50 imágenes', '',
           'Dos entrenamientos nuevos desde inicialización aleatoria con 50 imágenes, semillas 20261124 y 20261125. Se reutilizan y verifican los controles de 44 imágenes. Sin pesos externos, reanudación ni uso del test. Protocolo fijado antes de entrenar.', '',
           'Se evalúa last.pt a 1950 actualizaciones AdamW, con 3900 imágenes procesadas por ejecución. Inicialización idéntica dentro de cada pareja y distinta entre semillas. Batch2, lr0.0003 constante, stride8, propuestas×4 (RPN2000 en evaluación), aumentos fotométricos y muestreo original. No se aplican los filtros experimentales de inferencia.', '',
           '| Semilla | Imágenes | AP50 sin casco | TP/FP/FN a 0.05 | TP/FP/FN a 0.5 | AP50 con casco | mAP50 global |',
           '|---|---:|---:|---|---|---:|---:|']
    for r in records:
        a,b=r['counts05'][3],r['counts50'][3]
        lines.append(f"| {r['seed']} | {r['images']} | {100*r['metrics']['per_class'][3]['ap50']:.2f}% | {a['tp']}/{a['fp']}/{a['fn']} | {b['tp']}/{b['fp']}/{b['fn']} | {100*r['metrics']['per_class'][2]['ap50']:.2f}% | {100*r['metrics']['map50']:.2f}% |")
    lines+=['', 'TP: cabeza sin casco acertada; FP: falsa detección sin casco; FN: cabeza sin casco omitida. Emparejamiento uno a uno a IoU≥0.5. Cada ejecución evalúa las mismas seis cabezas sin casco y tres con casco.', '', '## Comparación por semilla', '']
    for seed in initial_by_seed:
        old,new=[r for r in records if r['seed']==seed]
        delta=100*(new['metrics']['per_class'][3]['ap50']-old['metrics']['per_class'][3]['ap50'])
        lines.append(f'- Semilla {seed}: AP50 sin casco {delta:+.2f} puntos.')
        for key,threshold in [('counts05','0.05'),('counts50','0.5')]:
            a,b=old[key][3],new[key][3]
            lines.append(f"  Confianza {threshold}: aciertos {a['tp']} → {b['tp']}; falsos positivos {a['fp']} → {b['fp']}; omisiones {a['fn']} → {b['fn']}.")
    lines+=['', '## Alcance y verificaciones', '',
            'Los 44 ejemplos anteriores se mantienen; se agregan seis escenas revisadas con siete cabezas reales y seis regiones no-cabeza confirmadas. Estas regiones no son una quinta clase ni tienen muestreo forzado. El experimento evalúa añadir las escenas completas, no permite atribuir causalmente un resultado solo a las regiones negativas.', '',
            'La validación conserva manifiesto, COCO e imágenes; se verifican los locks completos. Se comprobó el paso1950 de todos los estados AdamW, configuración, pesos iniciales, 3900 presentaciones y reproducción de las métricas guardadas. Los controles se identifican por SHA256 en protocol.json. El código del detector, cargador, métricas y monitor conserva el hash de los controles; train.py contiene soporte posterior para reserva de positivos, desactivado en estas ejecuciones.', '',
            'El dataset50 finaliza en época78; dataset44 en época89 parcial. Igual número de pasos y presentaciones no significa igual exposición por imagen (78 frente a 88.64 en promedio), composición de minibatches o frecuencia de evaluación por época.', '',
            'Cinco imágenes de validación repetidamente inspeccionadas y dos semillas no certifican generalización ni uso real. AP50 no es porcentaje de exactitud. No se escoge best.pt ni una semilla favorable y no se cambia el checkpoint de uso por defecto.', '',
            'Los JSON contienen AP de las cuatro clases, conteos a ambos umbrales, predicciones por imagen y hashes; las subcarpetas guardan configuración, pérdidas por componente, historial y diagnósticos de propuestas/cabezas.', '',
            'Reproducción: python training/run_fixed_updates.py --hard-negatives; después python training/report_hard_negative_training.py. Se requieren destinos nuevos para repetir, porque se rechaza sobrescribir ejecuciones o informes.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'records':records,'paired_deltas':pairs},indent=2))

if __name__=='__main__':main()
