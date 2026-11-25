"""Select a reproducible train-only annotation queue; no generated training labels."""
import base64
import argparse
import hashlib
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from training.association import iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--batch-id', default='train-separate-v2-01')
    parser.add_argument('--per-stratum', type=int, default=2)
    parser.add_argument('--focus-heads', action='store_true', help='Prioritize larger labeled helmet/no-helmet heads, mixed scenes and small heads')
    parser.add_argument('--focus-nohelmet', action='store_true', help='Train queue emphasizing source NoHelmet at varied sizes')
    parser.add_argument('--selection-file', type=Path, help='Explicit visually screened train candidates, still pending human annotation')
    parser.add_argument('--reuse-train-groups', action='store_true', help='Allow new images from groups already in training only')
    args = parser.parse_args()
    if args.reuse_train_groups and args.split != 'train':
        parser.error('Group reuse applies only to training')
    if args.selection_file and (args.focus_heads or args.focus_nohelmet):
        raise ValueError('Explicit selection cannot combine with other focus flags')
    if args.focus_nohelmet and (args.split != 'train' or args.focus_heads):
        raise ValueError('NoHelmet focus requires train and cannot combine with focus-heads')
    if Path(args.batch_id).name != args.batch_id or args.per_stratum < 1:
        raise ValueError('Invalid batch id or batch size')
    out = Path('data/annotations') / args.batch_id
    out.mkdir(parents=True, exist_ok=True)
    if (out/'draft.json').exists():
        raise FileExistsError('Batch already exists; do not overwrite annotation work')
    decisions = {r['image']: r for r in map(json.loads, Path('reports/preparation/image-decisions.jsonl').read_text().splitlines())}
    originals = list(map(json.loads, Path('reports/edgevision-manifest.jsonl').read_text().splitlines()))
    previous = [json.loads(p.read_text(encoding='utf-8')) for p in Path('data/annotations').glob('*/draft.json')]
    excluded_groups = {s['group_id'] for b in previous for s in b['scenes']}
    excluded_hashes = {s['sha256'] for b in previous for s in b['scenes']}
    if args.reuse_train_groups:
        excluded_groups = {s['group_id'] for b in previous for s in b['scenes'] if s['split'] != 'train'}
        excluded_groups.update(r['groupId'] for r in decisions.values() if r.get('split') in ('val','test'))
    if args.split == "test" and args.selection_file:
        # Existing test groups may supply new images, but never borrow a training or validation group.
        excluded_groups = {s["group_id"] for b in previous for s in b["scenes"] if s["split"] != "test"}
        excluded_groups.update(r["groupId"] for r in decisions.values() if r.get("split") in ("train", "val"))
    candidates = []
    for raw in originals:
        d = decisions.get(raw['image'], {})
        if d.get('split') != args.split or not d.get('detector') or d.get('groupId') in excluded_groups or raw['sha256'] in excluded_hashes:
            continue
        heads = [a for a in raw['annotations'] if a['class'] in ('Helmet', 'NoHelmet')]
        motos = [a for a in raw['annotations'] if a['class'] == 'BikeWithRider']
        if not heads or not motos or len(heads) > 5 or len(motos) > 4:
            continue
        scale = min(1., 640 / max(raw['width'], raw['height']))
        small = min(min(a['bbox'][2]-a['bbox'][0], a['bbox'][3]-a['bbox'][1])*scale for a in heads)
        overlap = max((iou(a['bbox'], b['bbox']) for i, a in enumerate(motos) for b in motos[i+1:]), default=0)
        mixed = len({a['class'] for a in heads}) == 2
        candidates.append((raw, d, small, overlap, mixed))
    selected = []; groups = set()
    # Four strata, two distinct split groups per stratum. Geometry is a proxy,
    # never a claim that a scarf/occlusion exists or that the annotation is right.
    strata = [('small_heads', lambda c: (c[2], c[0]['image'])),
              ('overlapping_contexts', lambda c: (-c[3], c[0]['image'])),
              ('mixed_helmet_labels', lambda c: (-int(c[4]), c[0]['image'])),
              ('baseline_examples', lambda c: (hashlib.sha256(c[0]['image'].encode()).hexdigest(),))]
    if args.focus_heads:
        if args.split != 'train': raise ValueError('Head-focused selection is for train only')
        def size_of_class(c, name):
            raw = c[0]; scale = min(1., 640/max(raw['width'],raw['height']))
            return max((min(a['bbox'][2]-a['bbox'][0],a['bbox'][3]-a['bbox'][1])*scale for a in raw['annotations'] if a['class']==name), default=0)
        strata = [('larger_helmet_source_labels', lambda c: (-size_of_class(c, 'Helmet'), c[0]['image'])),
                  ('larger_nohelmet_source_labels', lambda c: (-size_of_class(c, 'NoHelmet'), c[0]['image'])),
                  ('mixed_head_labels', lambda c: (-int(c[4]), -c[2], c[0]['image'])),
                  ('small_heads', lambda c: (c[2], c[0]['image']))]
    if args.focus_nohelmet:
        def nohelmet_sides(c):
            raw=c[0];scale=min(1.,640/max(raw['width'],raw['height']))
            return [min(a['bbox'][2]-a['bbox'][0],a['bbox'][3]-a['bbox'][1])*scale for a in raw['annotations'] if a['class']=='NoHelmet']
        strata=[('nohelmet_larger',lambda c:(-max(nohelmet_sides(c),default=0),c[0]['image'])),
                ('nohelmet_smaller',lambda c:(min(nohelmet_sides(c),default=float('inf')),c[0]['image'])),
                ('mixed_classes_context',lambda c:(-int(c[4]),-c[3],c[0]['image'])),
                ('diverse_reference',lambda c:(hashlib.sha256(('diversity-v3:'+c[0]['image']).encode()).hexdigest(),))]
    if args.split in ('val', 'test'):
        # Select validation independently of errors or model scores. This is
        # still a small eligible subset, not a representative benchmark.
        prefix = 'validation-v2:' if args.split == 'val' else 'test-v2:'
        strata = [('fixed_hash_validation' if args.split == 'val' else 'fixed_hash_test', lambda c: (hashlib.sha256((prefix+c[0]['image']).encode()).hexdigest(),))]
    for reason, key in strata:
        count = 0
        for c in sorted(candidates, key=key):
            if reason.startswith('nohelmet_') and not nohelmet_sides(c):
                continue
            if c[1]['groupId'] in groups:
                continue
            selected.append((c, reason)); groups.add(c[1]['groupId']); count += 1
            if count == args.per_stratum:
                break
    if not args.selection_file and len(selected) != len(strata)*args.per_stratum:
        raise ValueError('Not enough unused groups; reduce batch size explicitly')
    if args.selection_file:
        requested = json.loads(args.selection_file.read_text(encoding='utf-8'))['selected']
        by_id = {Path(c[0]['image']).stem: c for c in candidates}
        selected = []; groups = set(); selected_ids = set()
        for item in requested:
            c = by_id.get(item['image_id'])
            if c is None or c[1]['groupId'] in groups or item['image_id'] in selected_ids:
                raise ValueError('Ineligible, duplicate or previously used selection: '+item['image_id'])
            if not isinstance(item.get('reason'), str) or not item['reason'].strip():
                raise ValueError('Selection reason required')
            selected.append((c, item['reason']))
            groups.add(c[1]['groupId']); selected_ids.add(item['image_id'])
        if not selected: raise ValueError('Empty explicit selection')
    scenes = []; cards = []
    for (raw, d, small, overlap, mixed), reason in selected:
        entities = []; links = []; mi = hi = 0
        for index, ann in enumerate(raw['annotations']):
            origin = {'source_annotation_index': index, 'class': ann['class']}
            if ann['class'] == 'BikeWithRider':
                entities.append({'id': f'm{mi}', 'kind': 'motorcycle', 'bbox': None, 'context_bbox': ann['bbox'], 'origin': origin, 'review_status': 'pending'})
                mi += 1
            elif ann['class'] in ('Helmet', 'NoHelmet'):
                entities += [{'id': f'h{hi}', 'kind': 'head', 'bbox': ann['bbox'], 'helmet': 'con_casco' if ann['class'] == 'Helmet' else 'sin_casco', 'origin': origin, 'review_status': 'pending'},
                             {'id': f'o{hi}', 'kind': 'occupant', 'bbox': None, 'role': 'unknown', 'review_status': 'pending'}]
                links.append({'head_id': f'h{hi}', 'person_id': f'o{hi}', 'motorcycle_id': None, 'status': 'pending'})
                hi += 1
        scene = {'image_id': Path(raw['image']).stem, 'image': raw['image'], 'sha256': raw['sha256'], 'width': raw['width'], 'height': raw['height'],
                 'split': args.split, 'group_id': d['groupId'], 'selection_reason': reason,
                 'selection_features': {'min_head_side_at_640': small, 'max_context_iou': overlap, 'mixed_source_labels': mixed},
                 'note': ('ENTRENAMIENTO. ' if args.split == 'train' else 'EVALUACIÓN '+args.split.upper()+': no trasladar a entrenamiento. ')+'Revisar etiquetas originales, dibujar moto y persona separadas y elegir la moto de cada ocupante. Si faltan entidades, no declarar cobertura completa; solicitar ampliar el lote.',
                 'entities': entities, 'links': links, 'reviewed': False, 'annotation_complete': False}
        scenes.append(scene)
        with Image.open(raw['image']) as im:
            thumb = ImageOps.contain(im.convert('RGB'), (380, 320))
        card = Image.new('RGB', (400, 360), 'white'); card.paste(thumb, ((400-thumb.width)//2, 35))
        ImageDraw.Draw(card).text((8, 8), scene['image_id']+' / '+reason, fill='black'); cards.append(card)
    batch = {'schema_version': 1, 'batch_id': args.batch_id, 'provenance': 'existing_source_annotations_only; no model-generated labels',
             'human_reviewed': False, 'training_ready': False, 'held_out_usage': f'Original {args.split} only; preserve groups and splits.',
             'excluded_prior_batch_ids': [b['batch_id'] for b in previous], 'scenes': scenes}
    if args.selection_file:
        batch['selection_file'] = str(args.selection_file)
        batch['selection_sha256'] = hashlib.sha256(args.selection_file.read_bytes()).hexdigest()
    (out/'draft.json').write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding='utf-8')
    for scene in scenes:
        with Image.open(scene['image']) as im:
            im = im.convert('RGB'); im.thumbnail((1200, 1200)); buf = io.BytesIO(); im.save(buf, format='JPEG', quality=90)
        scene['preview'] = 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()
    template = Path('training/annotation-review.html').read_text(encoding='utf-8')
    template = template.replace('<h3>3. Revisión</h3>', '<h3>3. Revisión</h3><label><input type="checkbox" id="complete">Están representadas todas las motos y personas ocupantes visibles; no faltan entidades. Si faltan, dejar desmarcado.</label>')
    template = template.replace("$('reviewed').checked=s.reviewed;", "$('reviewed').checked=s.reviewed;$('complete').checked=s.annotation_complete;")
    template = template.replace("$('save').onclick", "$('complete').onchange=e=>{scene().annotation_complete=e.target.checked;scene().reviewed=false;$('reviewed').checked=false;};$('save').onclick")
    template = template.replace("'relations-pilot-reviewed.json':'relations-pilot-draft.json'", f"'{args.batch_id}-reviewed.json':'{args.batch_id}-draft.json'")
    (out/'review.html').write_text(template.replace('/*BATCH_DATA*/', json.dumps(batch, ensure_ascii=False).replace('<', '\\u003c')), encoding='utf-8')
    sheet = Image.new('RGB', (1600, 360*((len(cards)+3)//4)), '#dddddd')
    for i, card in enumerate(cards): sheet.paste(card, ((i%4)*400, (i//4)*360))
    sheet.save(out/'contact-sheet.jpg')
    print(json.dumps({'images': len(scenes), 'groups': len(groups), 'entities': sum(len(s['entities']) for s in scenes), 'links': sum(len(s['links']) for s in scenes), 'output': str(out)}))


if __name__ == '__main__': main()
