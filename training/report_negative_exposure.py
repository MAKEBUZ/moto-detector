"""Audit and summarize actual per-update negative-region exposure logs."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarize(rows,threshold):
    values=[r['thresholds'][threshold] for r in rows]
    return {'presentations':len(rows),
            'without_background_candidates':sum(v['available']['0']==0 for v in values),
            'with_candidates_not_sampled':sum(v['available']['0']>0 and v['selected']['0']==0 for v in values),
            'with_background_selected':sum(v['selected']['0']>0 for v in values),
            'selected_by_label':{str(k):sum(v['selected'][str(k)] for v in values) for k in (-1,0,1,2,3,4)},
            'available_by_label':{str(k):sum(v['available'][str(k)] for v in values) for k in (-1,0,1,2,3,4)}}


def main():
    out=Path('reports/training/negative-exposure-02');read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    protocol=read(out/'protocol.json');comparison=read(out/'comparison.json');summaries=[]
    manifest=[json.loads(l) for l in Path('data/prepared/helmet-separate-v2-expanded-04/train-manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    regions={r['image_id']:r['reviewed_nonhead_regions'] for r in manifest if r.get('reviewed_nonhead_regions')};assert len(regions)==6
    for job in protocol['jobs']:
        path=Path('runs')/job['run']/'negative-exposure.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()]
        assert len(rows)==936
        keys=[(r['update'],r['image_id'],r['region_id'],r['stage']) for r in rows];assert len(set(keys))==len(keys)
        counts=Counter((r['image_id'],r['stage']) for r in rows)
        assert counts==Counter({(image,stage):78 for image in regions for stage in ('rpn','roi')})
        for image in regions:
            for stage in ('rpn','roi'):
                assert sorted(r['epoch'] for r in rows if r['image_id']==image and r['stage']==stage)==list(range(1,79))
        for r in rows:
            assert r['image_id'] in regions and r['region_id']=='n0' and 1<=r['update']<=1950
            assert (r['update'],r['image_id'],r['region_id'],'roi' if r['stage']=='rpn' else 'rpn') in set(keys)
            quota=256 if r['stage']=='rpn' else 512
            assert r['total_selected_positive']+r['total_selected_negative']<=quota
            for v in r['thresholds'].values():
                assert len(v['selected_indices'])==len(set(v['selected_indices']))==sum(v['selected'].values())
                assert v['selected']['-1']==0
                assert all(0<=v['selected'][str(k)]<=v['available'][str(k)] for k in (-1,0,1,2,3,4))
                assert v['selected']['0']<=r['total_selected_negative']
                assert sum(v['selected'][str(k)] for k in (1,2,3,4))<=r['total_selected_positive']
        totals={stage:{t:summarize([r for r in rows if r['stage']==stage],t) for t in ('0.3','0.5')} for stage in ('rpn','roi')}
        per_region={image:{stage:{t:summarize([r for r in rows if r['image_id']==image and r['stage']==stage],t) for t in ('0.3','0.5')} for stage in ('rpn','roi')} for image in regions}
        phases={str(lo)+'-'+str(hi):{stage:{t:summarize([r for r in rows if lo<=r['epoch']<=hi and r['stage']==stage],t) for t in ('0.3','0.5')} for stage in ('rpn','roi')} for lo,hi in ((1,26),(27,52),(53,78))}
        report={**job,'log_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'log_records':len(rows),'totals':totals,'per_region':per_region,'phases':phases}
        summaries.append(report)
        with gzip.open(out/(job['run']+'-exposure.jsonl.gz'),'wb') as f:f.write(path.read_bytes())
    (out/'exposure-summary.json').write_text(json.dumps({'runs':summaries,'historical_actual_samples':True},indent=2),encoding='utf-8')
    lines=['# Registro real de exposición a negativos difíciles','',
           'Dos entrenamientos completos nuevos desde inicialización aleatoria, 50 imágenes, semillas20261124/20261125, 1950 actualizaciones y3900 presentaciones cada uno. Muestreador original; sin reserva de positivos/negativos, sin pesos externos y sin reanudar checkpoints. Se registraron las selecciones que realmente calcularon las pérdidas, después de los aumentos y del cambio de tamaño.', '',
           'Cada región apareció78 veces por etapa;936 filas por semilla (6 regiones ×78 épocas ×2 etapas). Los índices son locales a los candidatos de esa imagen/actualización, no identificadores persistentes. RPN:1 significa objeto,0 fondo,-1 ignorado. RoI:1 moto,2 persona,3 casco,4 sin casco y0 fondo.', '',
           '## Exposición real (IoU≥0.5 con región revisada)', '',
           '| Semilla | Etapa | Presentaciones | Sin candidatos fondo | Con candidatos, ninguno seleccionado | Con fondo seleccionado | Total candidatos fondo seleccionados |',
           '|---|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        for stage in ('rpn','roi'):
            t=s['totals'][stage]['0.5'];lines.append(f"| {s['seed']} | {stage} | {t['presentations']} | {t['without_background_candidates']} | {t['with_candidates_not_sampled']} | {t['with_background_selected']} | {t['selected_by_label']['0']} |")
    lines+=['','## Por región: presentaciones con fondo seleccionado /78 (IoU≥0.5)','', '| Imagen | Semilla | RPN | RoI | RoI sin candidatos de fondo |', '|---|---|---:|---:|---:|']
    for s in summaries:
        for image,v in s['per_region'].items():
            lines.append(f"| {image} | {s['seed']} | {v['rpn']['0.5']['with_background_selected']} | {v['roi']['0.5']['with_background_selected']} | {v['roi']['0.5']['without_background_candidates']} |")
    lines+=['','## Modelos finales frente a repeticiones anteriores sin registro','', '| Semilla | Ejecución | AP50 sin casco | TP/FP/FN a0.05 | TP/FP/FN a0.5 | mAP50 global |','|---|---|---:|---|---|---:|']
    for r in comparison['records']:
        a,b=r['counts05'][3],r['counts50'][3];kind='registro real' if 'exposure2' in r['run'] else 'anterior sin registro'
        lines.append(f"| {r['seed']} | {kind} | {r['metrics']['per_class'][3]['ap50']*100:.2f}% | {a['tp']}/{a['fp']}/{a['fn']} | {b['tp']}/{b['fp']}/{b['fn']} | {r['metrics']['map50']*100:.2f}% |")
    lines+=['','## Reproducibilidad y límites','',
            'La prueba CPU conserva exactamente pérdidas, gradientes, buffers y RNG; la prueba GPU con la misma entrada conserva exactamente pérdidas, RNG CPU/CUDA y máscaras de selección RPN/RoI. Se comprueba también que la metadata sigue el volteo y redimensionamiento sin cambiar aumentos. No se hacen sorteos adicionales ni se alteran las máscaras del muestreador.', '',
            'Las ejecuciones completas CUDA no son reproducibles bit a bit con esta configuración: dos pruebas sin observador a dos pasos ya difieren. Las pérdidas fueron1.739662 y1.737720; con observador1.736006. No atribuir diferencias frente a checkpoints históricos al registro ni tratar estas repeticiones como experimento causal de una mejora. Se conserva el diagnóstico en ../negative-exposure-01/gpu-equivalence-investigation.json. La primera ejecución larga se interrumpió por esta investigación; sus resultados parciales quedan excluidos. Las dos ejecuciones aquí parten de cero.', '',
            'La validación permanece intacta; no se usa test. El registro no vuelve a ejecutar el muestreador ni garantiza selección de las regiones. Las cajas revisadas no son necesariamente fondo para moto/persona: se registran también positivos e ignorados. Hay sensibilidad aIoU0.3 y desglose temporal por tercios en exposure-summary.json. Los registros completos comprimidos están junto a este informe.', '',
            'Reproducción: training/run_negative_exposure.py; training/report_negative_exposure_models.py; training/report_negative_exposure.py. No sobrescribir ejecuciones existentes.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps([{'seed':s['seed'],'totals':s['totals']} for s in summaries],indent=2))


if __name__=='__main__':main()
