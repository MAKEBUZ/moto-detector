"""Export only fully reviewed, complete train scenes into a new staging manifest."""
import argparse
import hashlib
import json
from pathlib import Path
from training.validate_annotation_batch import validate

CLASSES = ['motocicleta', 'persona_ocupante', 'cabeza_con_casco', 'cabeza_sin_casco']


def convert_scene(scene, expected_split='train'):
    if expected_split not in ('train', 'val', 'test') or scene['split'] != expected_split:
        raise ValueError('Only original requested split allowed')
    if scene.get('annotation_complete') is not True:
        raise ValueError('Human coverage declaration missing')
    if any(e['review_status'] in ('pending', 'uncertain') for e in scene['entities']):
        raise ValueError('Unresolved entities: exclude whole image, not just their boxes')
    if any(l['status'] in ('pending', 'uncertain') for l in scene['links']):
        raise ValueError('Unresolved relations')
    accepted = [e for e in scene['entities'] if e['review_status'] == 'confirmed']
    connected = {l[k] for l in scene['links'] if l['status'] == 'confirmed' for k in ('head_id', 'person_id')}
    if any(e['id'] not in connected for e in accepted if e['kind'] in ('head', 'occupant')):
        raise ValueError('Confirmed head/person without confirmed relation')
    annotations = []
    for e in accepted:
        if e['kind'] == 'motorcycle': cls = 0
        elif e['kind'] == 'occupant': cls = 1
        elif e['kind'] == 'head' and e['helmet'] in ('con_casco', 'sin_casco'):
            cls = 2 if e['helmet'] == 'con_casco' else 3
        else: raise ValueError('Unsupported or unknown entity class')
        annotations.append({'entity_id': e['id'], 'classId': cls, 'box': e['bbox']})
    if not annotations: raise ValueError('No confirmed annotations')
    return {k: scene[k] for k in ('image_id', 'image', 'sha256', 'width', 'height', 'split', 'group_id')} | {'annotations': annotations, 'links': [l for l in scene['links'] if l['status'] == 'confirmed']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviewed', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory; never overwrite')
    p.add_argument('--canonical', type=Path, default=Path('data/annotations/train-separate-v2-01/draft.json'))
    p.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    a = p.parse_args()
    canonical = json.loads(a.canonical.read_text(encoding='utf-8'))
    batch = json.loads(a.reviewed.read_text(encoding='utf-8'))
    check = validate(batch, canonical, reviewed=True)
    if not check['valid']: raise ValueError(check['errors'])
    decisions = {r['image']: r for r in map(json.loads, Path('reports/preparation/image-decisions.jsonl').read_text().splitlines())}
    rows = []; excluded = []
    for scene in batch['scenes']:
        d = decisions[scene['image']]
        if d['split'] != a.split or d['groupId'] != scene['group_id'] or d['sha256'] != scene['sha256']:
            raise ValueError('Original split/group/hash mismatch')
        try: rows.append(convert_scene(scene, a.split))
        except ValueError as exc: excluded.append({'image_id': scene['image_id'], 'reason': str(exc)})
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/f'{a.split}-manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    report = {'classes': CLASSES, 'split': a.split, 'accepted_images': len(rows), 'excluded': excluded,
              'reviewed_sha256': hashlib.sha256(a.reviewed.read_bytes()).hexdigest(), 'training_ready': False,
              'next_requirement': 'Expand reviewed train dataset and prepare independent validation with the same four classes; adapt model head and loader; initialize all weights randomly.'}
    (a.output/'preparation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
