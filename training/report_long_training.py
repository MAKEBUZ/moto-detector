"""Summarize the longer from-scratch experiment without changing selection."""
import hashlib
import json
import shutil
from pathlib import Path


def main():
    run=Path('runs/detector-separate-v2-long-01')
    out=Path('reports/training/separate-v2-long-01'); out.mkdir(parents=True,exist_ok=True)
    config=json.loads((run/'config.json').read_text())
    result=json.loads((run/'result.json').read_text())
    history=json.loads((run/'history.json').read_text())
    best=history[result['best_epoch']-1]
    previous=Path('runs/detector-separate-v2-expanded-01')
    prior_result=json.loads((previous/'result.json').read_text())
    prior=json.loads((previous/'history.json').read_text())[prior_result['best_epoch']-1]
    for name in ('config.json','result.json','history.json','requirements.lock.txt'):
        shutil.copyfile(run/name,out/name)
    shutil.copytree(run/'head-probes',out/'head-probes',dirs_exist_ok=True)
    first_heads=next((h['epoch'] for h in history if any(c['ap50']>0 for c in h['validation']['per_class'][2:])),None)
    report={'previous_best':prior,'long_best':best,'last':history[-1],'first_nonzero_head_validation_epoch':first_heads,
            'checkpoint_sha256':{n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in ('best.pt','last.pt')},
            'steps':150*13,'test_used':False,'selection_metric':'overall validation map50'}
    (out/'comparison.json').write_text(json.dumps(report,indent=2))
    lines=['# Entrenamiento prolongado desde cero','',
           '25 imágenes de entrenamiento, 5 de validación originales; 150 épocas (1950 actualizaciones). Pesos aleatorios, semilla 20261124, batch 2, AdamW lr constante 0.0003. Se conservaron arquitectura y aumentos. No se cargaron pesos anteriores ni externos. Test no se utilizó.', '',
           'La huella inicial coincide con la de los experimentos anteriores y el manifiesto de validación permanece idéntico. Se modificaron duración y planificación del aprendizaje respecto a las 20 épocas con coseno; no se atribuye causalidad a un único cambio.', '',
           f"Mejor checkpoint por mAP global: época {result['best_epoch']}. Primera época con AP de cabeza positivo en validación: {first_heads}. Duración: {result['seconds']:.1f} segundos.", '',
           '| AP@0.50 | Anterior, 20 épocas | Prolongado, mejor checkpoint |', '|---|---:|---:|']
    for name,a,b in zip(config['classes'],prior['validation']['per_class'],best['validation']['per_class']):
        lines.append(f"| {name} | {100*a['ap50']:.2f} % | {100*b['ap50']:.2f} % |")
    lines += [f"| Promedio | {100*prior['validation']['map50']:.2f} % | {100*best['validation']['map50']:.2f} % |",'',
              '## Seguimiento de cabezas', '',
              '| Época | RPN train /45 | RPN val /9 | Cabezas con clase correcta val /9, confianza ≥0.5 | Clasificación con caja real val /9 |', '|---|---:|---:|---:|---:|']
    for h in history:
        if 'head_monitor' not in h:continue
        t=h['head_monitor']['train'];v=h['head_monitor']['val']
        lines.append(f"| {h['epoch']} | {t['rpn_covered_iou50']} | {v['rpn_covered_iou50']} | {v['correct_head_coverage_iou50_score50']} | {v['oracle_correct']} |")
    lines += ['', 'Cobertura a IoU ≥0.50 por referencia; no equivale a recall con emparejamiento uno a uno. La clasificación con caja real es una prueba diagnóstica con etiquetas conocidas, no inferencia normal. El AP usa emparejamiento uno a uno y las salidas normales del detector; no es exactitud ni COCO mAP50:95.', '',
              '## Pérdidas por componente', '', '| Componente | Época 1 | Época 150 |', '|---|---:|---:|']
    for k,v in history[0]['train_loss_components'].items():
        lines.append(f"| {k} | {v:.6f} | {history[-1]['train_loss_components'][k]:.6f} |")
    lines += ['', 'Las pérdidas incluyen las cuatro clases, no solo cabezas. Los diagnósticos se guardaron cada diez épocas (además de la primera), con estado aleatorio preservado para no alterar el muestreo de entrenamiento.', '',
              '## Límites y artefactos', '',
              'Cinco imágenes y nueve cabezas de validación son una muestra pequeña, utilizada repetidamente para seleccionar checkpoints. No constituye una evaluación independiente final. No se habilitan conteo, multas ni asociación de cuatro clases.', '',
              '`history.json` contiene métricas y pérdidas por época; `head-probes/` contiene propuestas y diagnósticos detallados. Los checkpoints permanecen en `runs/detector-separate-v2-long-01/`. `comparison.json` incluye hashes y métricas. Las versiones anteriores permanecen intactas.', '',
              'Reproducción: ejecutar training/train.py con --task detector --separate-classes --dataset data/prepared/helmet-separate-v2-expanded-01 --epochs 150 --batch-size 2 --workers 0 --seed 20261124 --constant-lr --head-probe-every 10 y un --run nuevo. No hay reanudación implícita.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'best_epoch':result['best_epoch'],'best_validation':best['validation'],'last_monitor':history[-1]['head_monitor'],'first_heads':first_heads},indent=2))


if __name__=='__main__':main()
