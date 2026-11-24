"""Locate divergences in two fresh same-seed GPU training traces."""
import gzip
import hashlib
import json
from pathlib import Path
import torch


def tensor_differences(a,b):
    assert list(a)==list(b)
    result=[]
    for name in a:
        x,y=a[name],b[name];assert x.shape==y.shape and x.dtype==y.dtype
        if not torch.equal(x,y):
            d=(x.double()-y.double()).abs()
            result.append({'name':name,'different_elements':int((x!=y).sum()),'elements':x.numel(),
                           'max_abs':float(d.max()),'mean_abs':float(d.mean())})
    return {'different_tensors':len(result),'total_tensors':len(a),
            'max_abs':max((r['max_abs'] for r in result),default=0.),'details':result}


def main():
    out=Path('reports/training/gpu-reproducibility-01');read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    protocol=read(out/'protocol.json');jobs=protocol['jobs'];assert len(jobs)==2
    folders=[Path('runs')/j['run'] for j in jobs]
    traces=[[json.loads(l) for l in (f/'reproducibility.jsonl').read_text().splitlines()] for f in folders]
    configs=[read(f/'config.json') for f in folders]
    for rows in traces:assert [r['update'] for r in rows]==list(range(1,1951))
    assert configs[0]['initial_state_sha256']==configs[1]['initial_state_sha256']
    assert configs[0]['gpu_reproducibility_settings']==configs[1]['gpu_reproducibility_settings']
    for name in ('train.py','models.py','data.py','negative_exposure.py','reproducibility_trace.py','metrics.py','overfit_heads.py'):
        key=str(Path('training')/name);assert configs[0]['source_sha256'][key]==configs[1]['source_sha256'][key]
    fields=['image_indices','inputs_sha256','targets_sha256','rng_before','losses','samplers','rng_after_forward','rng_after_step','gradient_sha256','model_sha256']
    differences={}
    for key in fields:
        comparable=[(a,b) for a,b in zip(*traces) if key in a and key in b]
        mismatches=[a['update'] for a,b in comparable if a[key]!=b[key]]
        differences[key]={'compared_steps':len(comparable),'different_steps':len(mismatches),'first_difference':mismatches[0] if mismatches else None}
    sampler_fields={}
    for stage in ('rpn','roi'):
        for key in ('labels_sha256','positive_sha256','negative_sha256','positive_counts','negative_counts'):
            steps=[]
            for a,b in zip(*traces):
                left=next(r for r in a['samplers'] if r['stage']==stage);right=next(r for r in b['samplers'] if r['stage']==stage)
                if left[key]!=right[key]:steps.append(a['update'])
            sampler_fields[stage+'.'+key]={'different_steps':len(steps),'first_difference':steps[0] if steps else None}
    first=[torch.load(f/'first-step.pt',map_location='cpu',weights_only=True) for f in folders]
    first_differences={k:tensor_differences(first[0][k],first[1][k]) for k in ('model','clipped_gradients')}
    final=[torch.load(f/'last.pt',map_location='cpu',weights_only=True) for f in folders]
    final_differences=tensor_differences(final[0]['model'],final[1]['model'])
    assert all(c['optimizer_updates']==1950 and c['image_presentations']==3900 for c in final)
    comparison=read(out/'comparison.json')
    logs={}
    for job,folder in zip(jobs,folders):
        logs[job['run']]={}
        for name in ('reproducibility.jsonl','negative-exposure.jsonl'):
            raw=(folder/name).read_bytes();logs[job['run']][name]=hashlib.sha256(raw).hexdigest()
            with gzip.open(out/(job['run']+'-'+name+'.gz'),'wb') as f:f.write(raw)
    summary={'same_initial_state':True,'same_source_code':True,'settings':configs[0]['gpu_reproducibility_settings'],
             'fields':differences,'sampler_fields':sampler_fields,'first_step_differences':first_differences,
             'final_model_differences':final_differences,'log_hashes':logs,'metrics':comparison['records']}
    (out/'reproducibility-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Reproducibilidad GPU: misma semilla, dos procesos nuevos','',
           'Semilla20261124, dataset50,1950 actualizaciones y3900 presentaciones por proceso. Inicialización aleatoria idéntica; sin pesos preentrenados, reanudación, reserva de positivos/negativos ni cambios al muestreador. Ambos procesos usan los mismos aumentos y registros pasivos. Configuración fijada antes de entrenar en protocol.json. Se mantuvieron las opciones actuales de GPU; no se activó un modo determinista nuevo.', '',
           '## Primera divergencia observada', '', '| Elemento | Primer paso distinto | Pasos distintos / comparados |','|---|---:|---:|']
    for key,value in differences.items():lines.append(f"| {key} | {value['first_difference'] if value['first_difference'] is not None else 'ninguno'} | {value['different_steps']}/{value['compared_steps']} |")
    lines+=['', 'Entradas, pérdidas, RNG y máscaras se registran en cada paso. Gradientes antes del recorte y estado del modelo se comparan en pasos1–10 y cada25; la tabla distingue primera divergencia observada de pasos no observados. Los RNG incluyen CPU, CUDA, Python y NumPy.', '',
            '## Magnitud numérica', '',
            f"- Gradientes después del recorte del primer paso: {first_differences['clipped_gradients']['different_tensors']} tensores distintos; diferencia absoluta máxima {first_differences['clipped_gradients']['max_abs']:.10g}.",
            f"- Modelo después del primer paso: {first_differences['model']['different_tensors']} tensores distintos; diferencia absoluta máxima {first_differences['model']['max_abs']:.10g}.",
            f"- Modelo final: {final_differences['different_tensors']} tensores distintos; diferencia absoluta máxima {final_differences['max_abs']:.10g}.", '',
            '## Resultado final: last.pt, sin elegir mejor época', '', '| Repetición | AP50 sin casco | TP/FP/FN a0.05 | TP/FP/FN a0.5 | mAP50 global |','|---|---:|---|---|---:|']
    for r in comparison['records']:
        a,b=r['counts05'][3],r['counts50'][3]
        lines.append(f"| {r['repeat']} | {r['metrics']['per_class'][3]['ap50']*100:.2f}% | {a['tp']}/{a['fp']}/{a['fn']} | {b['tp']}/{b['fp']}/{b['fn']} | {r['metrics']['map50']*100:.2f}% |")
    lines+=['', 'AP50 es área bajo precisión/recall aIoU0.5, no exactitud. Validación original de cinco imágenes: seis cabezas sin casco y tres con casco. No se usa test ni se altera validación. Dos repeticiones detectan falta de reproducibilidad pero no estiman su distribución.', '',
            '## Configuración observada', '', '```json',json.dumps(summary['settings'],indent=2),'```', '',
            '## Verificación y artefactos', '',
            'Hashes de inicialización, datos, código relevante y configuración comparados; contador AdamW1950 y3900 presentaciones verificados. La reevaluación reproduce las métricas de ambos checkpoints. Las pruebas de observación pasiva conservan máscaras y RNG; no se modifica el entrenamiento para elegir un resultado favorable.', '',
            'reproducibility-summary.json detalla diferencias por tensor y por etapa del muestreador. Las trazas completas y los registros de exposición se conservan comprimidos aquí; first-step.pt y last.pt están en los directorios de ejecución. Modelo y gradientes del primer paso permiten separar diferencias en la propagación hacia atrás de diferencias posteriores del optimizador.', '',
            'Reproducción: training/run_gpu_reproducibility.py; training/report_gpu_reproducibility_models.py; training/report_gpu_reproducibility.py. Usar destinos nuevos; no sobrescribir el protocolo ni ejecuciones.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'fields':differences,'sampler_fields':sampler_fields,'first_gradient_max_abs':first_differences['clipped_gradients']['max_abs'],'final_model_max_abs':final_differences['max_abs']},indent=2))


if __name__=='__main__':main()
