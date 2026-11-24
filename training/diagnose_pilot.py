"""Inspect cached validation detections without tuning thresholds or weights."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from training.association import iou
from training.validate_annotation_batch import DEFAULT, validate


def inspect_box(box, predictions, classes):
    candidates = [{'index': i, **p, 'iou': iou(box, p['box'])}
                  for i, p in enumerate(predictions) if p['class_id'] in classes]
    candidates.sort(key=lambda p: (-p['iou'], -p['score']))
    qualifying = [p for p in candidates if p['iou'] >= .5]
    active = [p for p in qualifying if p['score'] >= .5]
    status = 'localized' if active else 'below_confidence_threshold' if qualifying else 'no_matching_box_in_cache'
    return {'status': status, 'best_iou': candidates[0] if candidates else None,
            'matching_candidates': qualifying, 'nearby_candidates': candidates[:3]}


def main():
    reviewed = Path('data/annotations/relations-pilot-v2/relations-pilot-reviewed.json')
    cached = Path('reports/error-review/detector-validation.jsonl')
    comparison = Path('reports/relations-pilot/comparison.json')
    batch = json.loads(reviewed.read_text(encoding='utf-8'))
    canonical = json.loads(DEFAULT.read_text(encoding='utf-8'))
    check = validate(batch, canonical, reviewed=True)
    if not check['valid']:
        raise ValueError(check['errors'])
    previous = json.loads(comparison.read_text(encoding='utf-8'))
    for path in (reviewed, cached, DEFAULT):
        if previous['input_sha256'][str(path)] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Comparison is stale: ' + str(path))
    cache = {r['image']: r for r in map(json.loads, cached.read_text().splitlines())}
    out = Path('reports/relations-pilot/diagnostics'); out.mkdir(parents=True, exist_ok=True)
    cases = []; cards = []
    for scene, ref, result in zip(batch['scenes'], canonical['scenes'], previous['scenes']):
        entities = {e['id']: e for e in scene['entities']}
        originals = {e['id']: e for e in ref['entities']}
        predictions = cache[scene['image']]['predictions']
        for link in result['links']:
            if link['outcome'] == 'correct_candidate' and link['helmet_correct_if_localized'] is True:
                continue
            head = entities[link['head_id']]
            moto_box = originals[link['motorcycle_id']]['context_bbox']
            h = inspect_box(head['bbox'], predictions, {1, 2})
            m = inspect_box(moto_box, predictions, {0})
            active = [p for p in h['matching_candidates'] if p['score'] >= .5]
            conflicts = [{'first': a['index'], 'second': b['index'], 'mutual_iou': iou(a['box'], b['box'])}
                         for j, a in enumerate(active) for b in active[j+1:]
                         if a['class_id'] != b['class_id'] and iou(a['box'], b['box']) >= .6]
            case = {'image_id': scene['image_id'], 'head_id': head['id'], 'helmet_reference': head['helmet'],
                    'association_outcome': link['outcome'], 'head': h, 'motorcycle_context': m,
                    'opposing_active_head_pairs': conflicts}
            cases.append(case)
            with Image.open(scene['image']) as im:
                im = im.convert('RGB')
                scale = min(900/im.width, 800/im.height)
                im = im.resize((round(im.width*scale), round(im.height*scale)))
            draw = ImageDraw.Draw(im)
            def rectangle(box, color, label):
                b = [v*scale for v in box]
                draw.rectangle(b, outline=color, width=3)
                draw.text((b[0], max(0, b[1]-12)), label, fill=color, stroke_width=1, stroke_fill='black')
            rectangle(head['bbox'], '#00ff66', 'HUMAN '+head['id'])
            rectangle(moto_box, '#00ddff', 'CONTEXT '+link['motorcycle_id'])
            selected = {p['index']: p for endpoint in (h, m) for p in endpoint['nearby_candidates'] if p['iou'] > .05}
            for p in selected.values():
                rectangle(p['box'], '#ff5555' if p['score'] >= .5 else '#ffdd00', f"p{p['index']} c{p['class_id']} {p['score']:.3f} IoU {p['iou']:.2f}")
            filename = f"{scene['image_id']}-{head['id']}.jpg"
            im.save(out/filename)
            case['preview'] = filename
            card = Image.new('RGB', (480, 500), 'white')
            thumb = ImageOps.contain(im, (480, 450)); card.paste(thumb, ((480-thumb.width)//2, 40))
            ImageDraw.Draw(card).text((8, 8), scene['image_id']+' '+head['id']+' / '+head['helmet'], fill='black')
            cards.append(card)
    sheet = Image.new('RGB', (480*3, 500*((len(cards)+2)//3)), '#dddddd')
    for i, card in enumerate(cards):
        sheet.paste(card, ((i%3)*480, (i//3)*500))
    sheet.save(out/'contact-sheet.jpg')
    report = {'confidence_threshold': .5, 'iou_threshold': .5, 'cache_scope': 'post-NMS detections already emitted by detector; not raw proposals',
              'thresholds_changed': False, 'weights_changed': False, 'test_used': False,
              'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (reviewed, cached, comparison, Path(__file__))}, 'cases': cases}
    (out/'diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps([{'image': c['image_id'], 'head': c['head_id'], 'head_status': c['head']['status'],
                       'motorcycle_status': c['motorcycle_context']['status'], 'conflict_pairs': len(c['opposing_active_head_pairs'])} for c in cases], indent=2))


if __name__ == '__main__':
    main()
