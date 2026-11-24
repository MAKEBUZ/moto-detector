"""Evaluate frozen geometric proposals against a reviewed, held-out pilot."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from training.association import associate, iou
from training.validate_annotation_batch import DEFAULT, validate


def match_boxes(predictions, truths, classes, threshold=.5, overlap=.5):
    """Confidence-ordered, one-to-one IoU matching; ignore helmet class."""
    result = {}
    used = set()
    order = sorted(range(len(predictions)), key=lambda i: (-predictions[i]['score'], i))
    for i in order:
        p = predictions[i]
        if p['class_id'] not in classes or p['score'] < threshold:
            continue
        candidates = [(iou(p['box'], t['bbox']), t['id']) for t in truths if t['id'] not in used]
        if candidates:
            score, identity = max(candidates)
            if score >= overlap:
                result[identity] = {'prediction_index': i, 'iou': score}
                used.add(identity)
    return result


def evaluate_scene(scene, reference, predictions):
    heads = [e for e in scene['entities'] if e['kind'] == 'head' and e['review_status'] == 'confirmed']
    accepted_motos = {e['id'] for e in scene['entities'] if e['kind'] == 'motorcycle' and e['review_status'] == 'confirmed'}
    # Preserve the old detector's box semantics; never match a rider-inclusive
    # prediction directly against a new motorcycle-only annotation.
    motos = [{'id': e['id'], 'bbox': e['context_bbox']} for e in reference['entities'] if e['id'] in accepted_motos]
    hm = match_boxes(predictions, heads, {1, 2})
    mm = match_boxes(predictions, motos, {0})
    associations = associate(predictions)
    by_head = {a['head_index']: a for a in associations['associations']}
    entities = {e['id']: e for e in scene['entities']}
    rows = []
    for link in scene['links']:
        if link['status'] != 'confirmed' or link['motorcycle_id'] is None:
            continue
        h = hm.get(link['head_id'])
        m = mm.get(link['motorcycle_id'])
        a = by_head.get(h['prediction_index']) if h else None
        if not h:
            outcome = 'head_not_localized'
        elif not m:
            outcome = 'motorcycle_context_not_localized'
        elif a['motorcycle_index'] is None:
            outcome = a['status']
        elif a['motorcycle_index'] == m['prediction_index']:
            outcome = 'correct_candidate'
        else:
            outcome = 'wrong_motorcycle_candidate'
        helmet = entities[link['head_id']]['helmet']
        helmet_ok = (predictions[h['prediction_index']]['class_id'] == {'con_casco': 1, 'sin_casco': 2}[helmet]) if h and helmet != 'unknown' else None
        rows.append({**link, 'outcome': outcome, 'head_match': h, 'motorcycle_context_match': m,
                     'helmet_correct_if_localized': helmet_ok, 'system_association': a})
    known_indices = {r['head_match']['prediction_index'] for r in rows if r['head_match']}
    return {'image_id': scene['image_id'], 'links': rows,
            'excluded_links': [l for l in scene['links'] if l['status'] != 'confirmed'],
            'other_system_heads_not_scored': [a for a in associations['associations'] if a['head_index'] not in known_indices],
            'occupancy_confirmed': False, 'counting_allowed': False}


def main():
    reviewed = Path('data/annotations/relations-pilot-v2/relations-pilot-reviewed.json')
    cache = Path('reports/error-review/detector-validation.jsonl')
    batch = json.loads(reviewed.read_text(encoding='utf-8'))
    canonical = json.loads(DEFAULT.read_text(encoding='utf-8'))
    validation = validate(batch, canonical, reviewed=True)
    if not validation['valid']:
        raise ValueError(validation['errors'])
    predictions = {r['image']: r for r in map(json.loads, cache.read_text().splitlines())}
    scenes = [evaluate_scene(s, ref, predictions[s['image']]['predictions']) for s, ref in zip(batch['scenes'], canonical['scenes'])]
    rows = [r for s in scenes for r in s['links']]
    outcomes = dict(Counter(r['outcome'] for r in rows))
    both = sum(bool(r['head_match'] and r['motorcycle_context_match']) for r in rows)
    correct = outcomes.get('correct_candidate', 0)
    summary = {'confirmed_reference_links': len(rows), 'outcomes': outcomes, 'both_endpoints_localized': both,
               'correct_candidates': correct, 'correct_candidate_fraction_all_reference_links': correct / len(rows) if rows else None,
               'correct_candidate_fraction_localized_endpoints': correct / both if both else None,
               'localized_heads': sum(r['head_match'] is not None for r in rows),
               'helmet_correct_on_localized_heads': sum(r['helmet_correct_if_localized'] is True for r in rows),
               'excluded_links': sum(len(s['excluded_links']) for s in scenes),
               'other_system_heads_not_scored': sum(len(s['other_system_heads_not_scored']) for s in scenes)}
    report = {'summary': summary, 'confidence_threshold': .5, 'matching_iou': .5,
              'matching': 'confidence-ordered one-to-one; head class agnostic; motorcycle uses canonical rider-inclusive context',
              'limitations': ['Selected seven-image validation pilot, not representative generalization accuracy.',
                              'No independent person detector; does not evaluate the full head-person-motorcycle chain.',
                              'Rejected/uncertain references and unmatched predictions are not scored as false positives.',
                              'Original combined motorcycle context boxes are proxies, not new human-confirmed pure motorcycle boxes.',
                              'Candidates remain unconfirmed; counting remains disabled.'],
              'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (reviewed, DEFAULT, cache, Path(__file__), Path('training/association.py'))},
              'test_used': False, 'retrained': False, 'scenes': scenes}
    out = Path('reports/relations-pilot'); out.mkdir(parents=True, exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    labels = {'correct_candidate': 'Vínculo candidato correcto', 'head_not_localized': 'Cabeza no localizada',
              'motorcycle_context_not_localized': 'Moto+ocupante no localizada', 'conflicting_head': 'Abstención por clases opuestas',
              'ambiguous': 'Abstención por ambigüedad', 'unassigned': 'Sin asignación geométrica',
              'duplicate_head': 'Cabeza duplicada', 'wrong_motorcycle_candidate': 'Candidato a moto incorrecta'}
    lines = ['# Comparación del piloto revisado', '',
             f"De {len(rows)} vínculos humanos confirmados, {correct} reciben el candidato correcto. En {both} se localizan ambos extremos.", '',
             'Confianza ≥ 0.5 e IoU ≥ 0.5; correspondencia uno a uno por confianza. El casco no condiciona la correspondencia de cabezas.', '',
             '| Imagen | Cabeza → moto | Resultado | Clase de casco correcta |', '|---|---|---|---|']
    for s in scenes:
        for r in s['links']:
            helmet = {True: 'Sí', False: 'No', None: 'No evaluable'}[r['helmet_correct_if_localized']]
            lines.append(f"| {s['image_id']} | {r['head_id']} → {r['motorcycle_id']} | {labels[r['outcome']]} | {helmet} |")
    lines += ['', '## Alcance', '',
              'La clase del detector es moto con ocupante. Se usa la caja de contexto original como referencia de identidad; las nuevas cajas de moto aislada no se comparan directamente con esa clase. No hay detector independiente de personas, por lo que no se evalúa la cadena completa cabeza→persona→moto.', '',
              f"Se excluyen {summary['excluded_links']} vínculos rechazados/inciertos. Hay {summary['other_system_heads_not_scored']} propuestas de cabeza adicionales fuera de los vínculos puntuados: no se consideran automáticamente falsos positivos. Rechazar una anotación no certifica un ejemplo negativo.", '',
              'Este piloto fue seleccionado por errores y no representa la precisión global. No se modificaron umbrales, pesos, anotaciones, particiones ni test. Los vínculos del sistema siguen siendo candidatos; el conteo permanece deshabilitado.', '',
              'Reproducir: `.venv/Scripts/python.exe -m training.evaluate_relations`. El JSON incluye resultados por vínculo y hashes de entradas y código.']
    (out/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
