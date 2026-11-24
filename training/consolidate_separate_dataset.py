"""Build an isolated four-class dataset from validated human review stages."""
import hashlib
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from PIL import Image
from training.prepare_separate_labels import CLASSES, convert_scene
from training.validate_annotation_batch import validate

SOURCES = [
    ('helmet-separate-v2-staging-01', 'train-separate-v2-01', 'train-separate-reviewed.json', 'train'),
    ('helmet-separate-v2-staging-02', 'train-separate-v2-02', 'train-separate-v2-02-reviewed.json', 'train'),
    ('helmet-separate-v2-validation-01', 'val-separate-v2-01', 'val-separate-v2-01-reviewed.json', 'val'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_disjoint(rows):
    for field in ('image_id', 'sha256'):
        values = [r[field] for r in rows]
        if len(values) != len(set(values)): raise ValueError('Duplicate '+field)
    train = {r['group_id'] for r in rows if r['split'] == 'train'}
    val = {r['group_id'] for r in rows if r['split'] == 'val'}
    if train & val: raise ValueError('Group leakage across splits')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-batch-03', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('data/prepared/helmet-separate-v2'))
    args = parser.parse_args()
    root = args.output
    sources = list(SOURCES)
    if args.include_batch_03:
        sources.append(('helmet-separate-v2-staging-03', 'train-separate-v2-03', 'train-separate-v2-03-reviewed.json', 'train'))
    if root.exists(): raise FileExistsError('Do not overwrite existing dataset')
    decisions = {r['image']: r for r in map(json.loads, Path('reports/preparation/image-decisions.jsonl').read_text().splitlines())}
    rows = []; excluded = []; provenance = {}
    for stage, batch, filename, split in sources:
        folder = Path('data/prepared')/stage
        reviewed = Path('data/annotations')/batch/filename
        draft = reviewed.parent/'draft.json'
        data = json.loads(reviewed.read_text(encoding='utf-8'))
        result = validate(data, json.loads(draft.read_text(encoding='utf-8')), reviewed=True)
        if not result['valid']: raise ValueError(result['errors'])
        prep = json.loads((folder/'preparation.json').read_text(encoding='utf-8'))
        if prep['classes'] != CLASSES or prep['reviewed_sha256'] != sha(reviewed):
            raise ValueError('Stale stage or class mismatch')
        manifest = folder/f'{split}-manifest.jsonl'
        accepted = list(map(json.loads, manifest.read_text(encoding='utf-8').splitlines()))
        scenes = {s['image_id']: s for s in data['scenes']}
        for row in accepted:
            if row != convert_scene(scenes[row['image_id']], split): raise ValueError('Stage differs from reviewed annotations')
            decision = decisions[row['image']]
            if (decision['split'], decision['groupId'], decision['sha256']) != (split, row['group_id'], row['sha256']):
                raise ValueError('Original partition mismatch')
            with Image.open(row['image']) as image:
                if image.size != (row['width'], row['height']): raise ValueError('Dimensions mismatch')
        rows.extend(accepted)
        excluded.extend({'batch_id': batch, **item} for item in prep['excluded'])
        provenance.update({str(p): sha(p) for p in (reviewed, draft, manifest, folder/'preparation.json')})
    assert_disjoint(rows)
    if any(not any(r['split'] == split for r in rows) for split in ('train', 'val')):
        raise ValueError('Empty split')
    root.mkdir(parents=True)
    summary = {}
    for split in ('train', 'val'):
        image_dir = root/'images'/split; image_dir.mkdir(parents=True)
        subset = []
        coco = {'images': [], 'annotations': [], 'categories': [{'id': i+1, 'name': name} for i, name in enumerate(CLASSES)]}
        for row in (r for r in rows if r['split'] == split):
            destination = image_dir/(row['image_id']+Path(row['image']).suffix)
            shutil.copyfile(row['image'], destination)
            if sha(destination) != row['sha256']: raise ValueError('Copied image hash mismatch')
            output = {**row, 'source_image': row['image'], 'image': destination.relative_to(root).as_posix()}
            subset.append(output)
            image_id = len(coco['images'])+1
            coco['images'].append({'id': image_id, 'file_name': output['image'], 'width': row['width'], 'height': row['height']})
            for ann in row['annotations']:
                x0, y0, x1, y1 = ann['box']
                coco['annotations'].append({'id': len(coco['annotations'])+1, 'image_id': image_id, 'category_id': ann['classId']+1,
                                            'bbox': [x0, y0, x1-x0, y1-y0], 'area': (x1-x0)*(y1-y0), 'iscrowd': 0})
        (root/f'{split}-manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in subset), encoding='utf-8')
        (root/f'{split}-coco.json').write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding='utf-8')
        counts = Counter(a['classId'] for r in subset for a in r['annotations'])
        summary[split] = {'images': len(subset), 'boxes': sum(counts.values()), 'links': sum(len(r['links']) for r in subset), 'classes': {name: counts[i] for i, name in enumerate(CLASSES)}}
    report = {'dataset': root.name, 'classes': CLASSES, 'summary': summary, 'excluded': excluded,
              'source_sha256': provenance, 'group_overlap_train_val': False, 'exact_image_overlap': False,
              'annotations_prepared': True, 'training_started': False, 'test_used': False,
              'limitation': 'Small human-reviewed subset; split groups do not certify independent scenes; use four-class loader and random initialization.'}
    (root/'dataset.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (root/'dataset.lock.json').write_text(json.dumps({p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}, indent=2), encoding='utf-8')
    report_dir = Path('reports/separate-v2')
    report_dir.mkdir(parents=True, exist_ok=True)
    report_name = root.name+'-prepared.json' if args.include_batch_03 else 'prepared-dataset.json'
    (report_dir/report_name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
