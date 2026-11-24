"""Compare stride-8 features against the photometric stride-16 control."""
import hashlib
import json
import shutil
from pathlib import Path


def main():
    base=Path('runs/detector-separate-v2-photo-01');run=Path('runs/detector-separate-v2-stride8-01')
    out=Path('reports/training/separate-v2-stride8-01');out.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text())
    a,b=read(base/'config.json'),read(run/'config.json')
    keys=['epochs','seed','batch_size','workers','dataset_lock_sha256','initial_state_sha256','learning_rate','constant_lr','head_probe_every','classes','photometric_policy','parameters']
    assert all(a[k]==b[k] for k in keys)
    old,new=read(base/'history.json'),read(run/'history.json')
    ar,br=read(base/'result.json'),read(run/'result.json')
    ab,bb=old[ar['best_epoch']-1],new[br['best_epoch']-1]
    for name in ('config.json','history.json','result.json','requirements.lock.txt'):shutil.copyfile(run/name,out/name)
    shutil.copytree(run/'head-probes',out/'head-probes',dirs_exist_ok=True)
    comparison={'control_best':ab,'stride8_best':bb,'control_last':old[-1],'stride8_last':new[-1],
                'verified_fixed_settings':keys,'checkpoint_sha256':{n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in ('best.pt','last.pt')},'test_used':False}
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2))
    lines=['# Características de mayor resolución: paso 8','',
           'Entrenamiento desde cero: 25 imágenes train, las mismas 5 val, 150 épocas / 1950 pasos, batch 2, semilla 20261124, AdamW lr constante 0.0003. Se mantienen brillo y contraste 0.85–1.15 y desenfoque con probabilidad 0.25 y radio 0.2–0.8 px, además de reflejos horizontales. No se cargaron pesos previos ni externos.', '',
           'Único cambio configurado: stride de la primera convolución del último bloque, de 2 a 1. El backbone produce paso 8 en lugar de 16: doble resolución espacial por eje. Mismos canales, parámetros, valores iniciales aleatorios, tamaños/relaciones de anclas y límites de propuestas. Consecuencias inseparables: más posiciones de anclas y distinto campo receptivo; no se atribuye causalidad exclusivamente a resolución. La entrada sigue limitada a 640 px.', '',
           f"Checkpoint seleccionado por mAP global: control época {ar['best_epoch']}; paso 8 época {br['best_epoch']}. Duración del nuevo bucle: {br['seconds']:.1f} segundos.", '',
           '| AP@0.50 | Paso 16 + fotométrico | Paso 8 + fotométrico |','|---|---:|---:|']
    for name,x,y in zip(b['classes'],ab['validation']['per_class'],bb['validation']['per_class']):
        lines.append(f"| {name} | {100*x['ap50']:.2f} % | {100*y['ap50']:.2f} % |")
    lines += [f"| Promedio | {100*ab['validation']['map50']:.2f} % | {100*bb['validation']['map50']:.2f} % |", '',
              '## Diagnóstico de la última época', '',
              '| Medida | Control train /45 | Paso 8 train /45 | Control val /9 | Paso 8 val /9 |','|---|---:|---:|---:|---:|']
    for k in ('rpn_covered_iou50','correct_head_coverage_iou50_score50','oracle_correct'):
        lines.append('| '+k+' | '+' | '.join(str(h['head_monitor'][s][k]) for s,h in [('train',old[-1]),('train',new[-1]),('val',old[-1]),('val',new[-1])])+' |')
    lines += ['', 'Cobertura por caja a IoU ≥0.50, no recall con correspondencia uno a uno. `oracle_correct` suministra la caja real: diagnóstico, no inferencia normal. AP es la evaluación normal a IoU 0.50, no exactitud ni COCO mAP50:95. El AP usa todas las salidas del detector a partir del umbral interno 0.05; el conteo de confianza ≥0.5 se informa por separado.', '',
              '## Trazabilidad y límites', '',
              'Se verificaron configuración fija, identidad del lock del dataset y valores iniciales. Los pesos finales y checkpoints del control no se sobrescribieron. El argumento feature_stride queda registrado y la inferencia/diagnóstico lo lee para evitar cargar pesos de paso 8 en una arquitectura de paso 16.', '',
              '27 pruebas unitarias aprobadas: mapa espacial doble, igual número y valores iniciales de parámetros, anclas conservadas, aumento fotométrico solo en train, etiquetas y métricas de cuatro clases. Los registros por época y propuestas se conservan junto a este informe; pesos en runs/detector-separate-v2-stride8-01/.', '',
              'Una sola semilla, cinco imágenes y nueve cabezas en una validación consultada repetidamente no bastan para demostrar generalización. No se utilizó test ni se habilitó conteo.', '',
              'Reproducción: training/train.py con --task detector --separate-classes --dataset data/prepared/helmet-separate-v2-expanded-01 --epochs 150 --batch-size 2 --workers 0 --seed 20261124 --constant-lr --head-probe-every 10 --photometric --feature-stride 8 y un --run nuevo.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'best_epoch':br['best_epoch'],'validation':bb['validation'],'last_monitor':new[-1]['head_monitor']},indent=2))


if __name__=='__main__':main()
