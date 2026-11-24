"""Compare four reserved-positive runs with their fixed-update controls."""
import gzip
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import torch
from data import SeparateHelmetDataset
from models import build_detector, forbid_external_weights
from metrics import detection_ap50
from evaluate_reserved import counts_at_threshold


@torch.inference_mode()
def main():
    out=Path('reports/training/object-reserve-01')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    protocol=read(out/'protocol.json')
    baseline_path=Path('reports/training/fixed-updates-01/comparison.json')
    controls={(r['images'],r['seed']):r for r in read(baseline_path)['records']}
    torch.set_num_threads(4);forbid_external_weights();device='cuda' if torch.cuda.is_available() else 'cpu'
    records=[]
    fixed=['dataset_lock_sha256','batch_size','workers','seed','epochs','max_updates','constant_lr',
           'learning_rate','photometric_policy','feature_stride','proposal_multiplier','head_probe_every',
           'classes','parameters','initial_state_sha256','initialization','pretrained']
    for job in protocol['jobs']:
        run=Path('runs')/job['run'];checkpoint=run/'last.pt'
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True);c=ck['config']
        old=controls[(job['images'],job['seed'])];old_config=read(Path('runs')/old['run']/'config.json')
        for key in fixed: assert c[key]==old_config[key],key
        assert c['rpn_object_reserve'] and not c['pretrained'] and c['initialization']=='random'
        assert ck['optimizer_updates']==1950 and {int(v['step']) for v in ck['optimizer']['state'].values()}=={1950}
        assert ck['image_presentations']==old['image_presentations']
        root=Path(job['dataset'])
        for name,digest in read(root/'dataset.lock.json').items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
        ds=SeparateHelmetDataset(root,'val',augment=False);samples=[ds[i] for i in range(len(ds))]
        model=build_detector(5,c['feature_stride'],c['proposal_multiplier'])
        model.load_state_dict(ck['model']);model.to(device).eval();pred=[]
        for start in range(0,len(samples),2):
            pred.extend([{k:v.cpu() for k,v in p.items()} for p in model([x.to(device) for x,_ in samples[start:start+2]])])
        targets=[t for _,t in samples];metrics=detection_ap50(pred,targets,(1,2,3,4))
        assert abs(metrics['map50']-ck['validation']['map50'])<1e-6
        # Inspect actual training selections, not a replay at final weights.
        by_class=defaultdict(Counter);updates=Counter();images=0
        log=run/'rpn-selections.jsonl'
        with log.open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line);images+=1;updates[row['update']]+=1
                assert row['positive_count']<=128 and row['positive_count']+row['negative_count']<=256
                assert sum(o['selected'] for o in row['objects'])==row['positive_count']
                assert sum(o['baseline_selected'] for o in row['objects'])==row['positive_count']
                indices=[]
                for obj in row['objects']:
                    assert obj['selected']==len(obj['anchor_indices'])<=obj['available']
                    assert obj['selected']>=1 or obj['available']==0
                    indices.extend(obj['anchor_indices'])
                    a=by_class[str(obj['class_id'])];a['object_presentations']+=1
                    a['available_positives']+=obj['available'];a['vanilla_selected']+=obj['baseline_selected'];a['reserved_selected']+=obj['selected']
                    a['no_available_positive']+=int(obj['available']==0)
                    a['eligible_missed_before']+=int(obj['available']>0 and obj['baseline_selected']==0)
                    a['eligible_missed_after']+=int(obj['available']>0 and obj['selected']==0)
                assert len(set(indices))==len(indices)
        assert images==ck['image_presentations'] and set(updates)==set(range(1,1951))
        assert all(1<=count<=2 for count in updates.values())
        rec={**job,'control_run':old['run'],'metrics':metrics,'counts50':counts_at_threshold(pred,targets,.5),
             'counts05':counts_at_threshold(pred,targets,.05),'control_metrics':old['metrics'],'control_counts50':old['counts50'],
             'nohelmet_ap_delta_points':100*(metrics['per_class'][3]['ap50']-old['metrics']['per_class'][3]['ap50']),
             'sampling_by_class':dict(by_class),'logged_image_presentations':images,
             'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             'selection_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
        records.append(rec);dest=out/job['run'];dest.mkdir(exist_ok=False)
        for name in ('config.json','history.json','result.json','requirements.lock.txt'):shutil.copyfile(run/name,dest/name)
        shutil.copytree(run/'head-probes',dest/'head-probes')
        with log.open('rb') as src,gzip.open(dest/'rpn-selections.jsonl.gz','wb') as dst:shutil.copyfileobj(src,dst)
        (dest/'predictions.json').write_text(json.dumps([{'image_id':r['image_id'],'prediction':{k:v.tolist() for k,v in p.items()},'target':{k:v.tolist() for k,v in t.items()}} for r,p,t in zip(ds.rows,pred,targets)],indent=2),encoding='utf-8')
        del model,ck
    (out/'comparison.json').write_text(json.dumps({'records':records,'fixed_settings_verified':fixed,
        'control_comparison_sha256':hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),'test_used':False},indent=2),encoding='utf-8')
    lines=['# Reserva de positivos RPN por objeto', '',
           'Cuatro entrenamientos nuevos desde cero: 35/44 imágenes y semillas 20261124/20261125. Cada uno termina exactamente en 1950 actualizaciones. Se compara last.pt con su control de idéntico dataset y semilla, no con el mejor checkpoint de otra ejecución. No se cargaron pesos externos ni se reanudaron modelos anteriores.', '',
           'Único cambio configurado de aprendizaje: si el muestreo RPN habitual deja un objeto elegible sin positivos, se sustituye un positivo redundante de otro objeto por uno del objeto omitido. Se conserva la cantidad total de positivos y negativos, así como las anclas negativas seleccionadas. La reparación usa un RNG independiente (semilla de ejecución +1000000), sin consumir el RNG global. No se cambian las etiquetas, anclas, pérdidas, RoI, presupuestos de propuestas, umbrales ni inferencia.', '',
           'La garantía solo aplica si el objeto tiene al menos un ancla asignada positiva. Si el presupuesto no alcanza para todos los objetos elegibles, se detiene con error; no se excede la cuota ni se oculta el caso. No ocurrió en estas ejecuciones.', '',
           '| Train | Semilla | AP sin casco control | Con reserva | Diferencia pp | TP/FP/FN a 0,5 control → reserva | mAP control → reserva |', '|---|---|---:|---:|---:|---|---|']
    for r in records:
        old=r['control_counts50'][3];new=r['counts50'][3]
        fmt=lambda v:'/'.join(str(v[k]) for k in ('tp','fp','fn'))
        lines.append(f"| {r['images']} | {r['seed']} | {r['control_metrics']['per_class'][3]['ap50']*100:.2f} % | {r['metrics']['per_class'][3]['ap50']*100:.2f} % | {r['nohelmet_ap_delta_points']:+.2f} | {fmt(old)} → {fmt(new)} | {r['control_metrics']['map50']*100:.2f} % → {r['metrics']['map50']*100:.2f} % |")
    lines+=['','## Selección real durante entrenamiento: sin casco','','Los conteos siguientes corresponden a presentaciones de objetos a lo largo de todos los pasos, no a cabezas únicas. «Antes» es el sorteo habitual dentro de esta ejecución justo antes de repararlo; no es una reconstrucción del registro de los controles históricos.','','| Train | Semilla | Presentaciones sin casco | Sin ancla positiva disponible | Omitidas antes | Omitidas después | Positivos seleccionados antes → después |','|---|---|---:|---:|---:|---:|---|']
    for r in records:
        v=r['sampling_by_class']['3']
        lines.append(f"| {r['images']} | {r['seed']} | {v['object_presentations']} | {v['no_available_positive']} | {v['eligible_missed_before']} | {v['eligible_missed_after']} | {v['vanilla_selected']} → {v['reserved_selected']} |")
    recovered=sum(r['sampling_by_class']['3']['eligible_missed_before'] for r in records)
    total=sum(r['sampling_by_class']['3']['object_presentations'] for r in records)
    lines+=['','## Interpretación y decisión','',
            f'La reserva corrigió {recovered} omisiones en {total} presentaciones de cabezas sin casco; ninguna elegible quedó sin positivo después de reparar. Son presentaciones repetidas, no imágenes ni personas distintas.', '',
            'La mejora de detección no es consistente: el AP sin casco sube en dos parejas, baja en una y empata en otra. El mAP global baja en las cuatro parejas. A confianza 0,5, tres ejecuciones siguen sin aciertos sin casco; la de 44 imágenes/semilla 20261125 consigue un acierto y dos falsos positivos. La de 44 imágenes/semilla 20261124 pierde el acierto que tenía su control.', '',
            'Conclusión: evitar omisiones de objetos en el muestreo RPN funciona como garantía técnica, pero no demuestra una mejora estable de generalización. Se conserva el flag experimental desactivado por defecto; no se reemplazan controles ni se habilita conteo.', '',
            'Antes de otro entrenamiento, comparar las etapas de ranking de propuestas y clasificación de estos checkpoints con sus controles, aprovechando los registros reales. No aumentar la reserva, el top-k o cambiar pérdidas simultáneamente para perseguir una mejora en esta validación pequeña.']
    lines+=['','## Verificaciones y alcance','','Se verificaron: configuración e inicialización iguales al control correspondiente, pesos aleatorios distintos entre semillas, 1950 pasos en los contadores AdamW, mismo número de presentaciones de imágenes por pareja y dataset intacto. La inferencia congelada reproduce las métricas guardadas. Las 34 pruebas incluyen reserva por objeto, presupuesto, negativos, ausencia de positivos, RNG global y equivalencia de inferencia antes/después de instalar el sampler. Una ejecución corta verificó integración y registros.', '',
            'Los archivos rpn-selections.jsonl.gz contienen, por paso e imagen, identificador de objeto, clase, positivos disponibles, sorteo anterior, selección final e índices de anclas. No guardan imágenes aumentadas ni coordenadas de todas las anclas, y no permiten por sí solos reconstruir toda la entrada aumentada.', '',
            'La validación sigue teniendo cinco imágenes reutilizadas: seis cabezas sin casco y tres con casco. Dos semillas no bastan para demostrar estabilidad estadística. Se conservaron los controles; test no se usó y el conteo sigue deshabilitado. AP50 no es exactitud ni mAP50:95. El cambio de positivos altera gradientes y trayectoria posterior; mantener semillas no implica trayectorias idénticas.', '',
            'Reproducción: `.venv/Scripts/python.exe training/run_fixed_updates.py --rpn-object-reserve`. Evaluación: `.venv/Scripts/python.exe training/report_object_reserve.py`. Las salidas existentes se rechazan.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('images','seed','nohelmet_ap_delta_points','counts50','sampling_by_class')} for r in records],indent=2))


if __name__=='__main__':main()
