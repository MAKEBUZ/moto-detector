"""Validate review overlays without changing datasets or certifying human identity."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

DEFAULT = Path('data/annotations/relations-pilot-v2/draft.json')
STATUSES = {'pending', 'confirmed', 'uncertain', 'rejected'}


def validate(batch, canonical, *, reviewed=False, verify_images=True):
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    check(batch.get('schema_version') == canonical['schema_version'], 'schema_version changed')
    check(batch.get('batch_id') == canonical['batch_id'], 'batch_id changed')
    check(batch.get('training_ready') is False, 'training_ready must remain false until dataset preparation')
    if reviewed:
        check(batch.get('human_reviewed') is True, 'human review declaration missing')
        check(isinstance(batch.get('reviewer'), str) and bool(batch['reviewer'].strip()), 'reviewer missing')
        try:
            stamp = datetime.fromisoformat(batch.get('reviewed_at') or '')
            check(stamp.tzinfo is not None, 'reviewed_at needs timezone')
        except (ValueError, TypeError):
            errors.append('invalid reviewed_at')
    scenes = batch.get('scenes', [])
    check([s.get('image_id') for s in scenes] == [s['image_id'] for s in canonical['scenes']], 'scene inventory changed')
    confirmed_links = []
    for s, ref in zip(scenes, canonical['scenes']):
        name = s.get('image_id', '?')
        for field in ('image_id', 'image', 'sha256', 'width', 'height', 'split', 'group_id'):
            check(s.get(field) == ref[field], f'{name}: immutable {field} changed')
        if verify_images:
            path = Path(ref['image'])
            check(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256'], f'{name}: source hash mismatch')
        if reviewed:
            check(s.get('reviewed') is True, f'{name}: scene not reviewed')
        entities = s.get('entities', [])
        check([(e.get('id'), e.get('kind')) for e in entities] == [(e['id'], e['kind']) for e in ref['entities']], f'{name}: entity inventory changed')
        ids = [e.get('id') for e in entities]
        check(len(ids) == len(set(ids)), f'{name}: duplicate entity id')
        mapping = {e.get('id'): e for e in entities}
        for e in entities:
            label = f'{name}/{e.get("id")}'
            status = e.get('review_status')
            check(status in STATUSES, f'{label}: invalid status')
            if reviewed:
                check(status != 'pending', f'{label}: pending entity')
            box = e.get('bbox')
            if box is not None:
                valid = isinstance(box, list) and len(box) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in box)
                check(valid, f'{label}: invalid box')
                if valid:
                    check(0 <= box[0] < box[2] <= ref['width'] and 0 <= box[1] < box[3] <= ref['height'], f'{label}: box outside image or degenerate')
            check(status != 'confirmed' or box is not None, f'{label}: confirmed entity needs box')
            if e.get('kind') == 'head':
                check(e.get('helmet') in {'con_casco', 'sin_casco', 'unknown'}, f'{label}: invalid helmet state')
        links = s.get('links', [])
        check([l.get('head_id') for l in links] == [l['head_id'] for l in ref['links']], f'{name}: link inventory changed')
        people = []
        for l in links:
            status = l.get('status')
            check(status in STATUSES, f'{name}: invalid link status')
            if reviewed:
                check(status != 'pending', f'{name}: pending link')
            h, p, m = (mapping.get(l.get(k)) for k in ('head_id', 'person_id', 'motorcycle_id'))
            check(h is not None and h.get('kind') == 'head', f'{name}: invalid head reference')
            check(p is not None and p.get('kind') in {'occupant', 'non_occupant'}, f'{name}: invalid person reference')
            check(l.get('motorcycle_id') is None or (m is not None and m.get('kind') == 'motorcycle'), f'{name}: invalid motorcycle reference')
            if status == 'confirmed':
                people.append(l.get('person_id'))
                check(all(e is not None and e.get('review_status') == 'confirmed' for e in (h, p)), f'{name}: unconfirmed head/person')
                if p and p.get('kind') == 'occupant':
                    check(m is not None and m.get('review_status') == 'confirmed', f'{name}: occupant needs confirmed motorcycle')
                if p and p.get('kind') == 'non_occupant':
                    check(l.get('motorcycle_id') is None, f'{name}: non-occupant assigned to motorcycle')
                confirmed_links.append({'image_id': name, **l})
        check(len(people) == len(set(people)), f'{name}: multiple heads assigned to same person')
    return {'valid': not errors, 'errors': errors, 'human_review_asserted': reviewed and batch.get('human_reviewed') is True,
            'training_ready': False, 'evaluation_scope': 'annotation structure only; not model accuracy or dataset readiness',
            'confirmed_links': confirmed_links if reviewed and not errors else []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical', type=Path, default=DEFAULT, help='Original draft for this batch')
    parser.add_argument('--reviewed', type=Path, help='JSON exported by the human reviewer')
    parser.add_argument('--output', type=Path, help='New validation receipt; existing files are never overwritten')
    args = parser.parse_args()
    canonical = json.loads(args.canonical.read_text(encoding='utf-8'))
    batch = json.loads((args.reviewed or args.canonical).read_text(encoding='utf-8'))
    try:
        result = validate(batch, canonical, reviewed=args.reviewed is not None)
    except (TypeError, KeyError, AttributeError, ValueError) as exc:
        result = {'valid': False, 'errors': [f'Malformed review: {exc}'], 'training_ready': False}
    result['input_sha256'] = hashlib.sha256((args.reviewed or args.canonical).read_bytes()).hexdigest()
    if args.output:
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
