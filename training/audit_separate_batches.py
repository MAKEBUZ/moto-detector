"""Verify new review queues against original split decisions and earlier batches."""
import json
from pathlib import Path
from training.validate_annotation_batch import validate


def main():
    decisions = {r['image']: r for r in map(json.loads, Path('reports/preparation/image-decisions.jsonl').read_text().splitlines())}
    paths = sorted(Path('data/annotations').glob('*/draft.json'))
    batches = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    targets = {'train-separate-v2-02': 'train', 'val-separate-v2-01': 'val'}
    summaries = []
    for batch in batches:
        if batch['batch_id'] not in targets: continue
        result = validate(batch, batch)
        if not result['valid']: raise ValueError(result['errors'])
        others = [s for b in batches if b['batch_id'] != batch['batch_id'] for s in b['scenes']]
        other_groups = {s['group_id'] for s in others}
        other_hashes = {s['sha256'] for s in others}
        scenes = batch['scenes']
        assert len({s['group_id'] for s in scenes}) == len(scenes)
        for s in scenes:
            d = decisions[s['image']]
            assert s['split'] == d['split'] == targets[batch['batch_id']]
            assert s['group_id'] == d['groupId'] and s['sha256'] == d['sha256']
            assert s['group_id'] not in other_groups and s['sha256'] not in other_hashes
        summaries.append({'batch_id': batch['batch_id'], 'split': targets[batch['batch_id']], 'images': len(scenes),
                          'entities': sum(len(s['entities']) for s in scenes), 'links': sum(len(s['links']) for s in scenes),
                          'original_hashes_verified': True, 'overlap_with_other_batches': False, 'human_reviewed': False})
    assert len(summaries) == len(targets)
    report = {'batches': summaries, 'classes': ['motocicleta', 'persona_ocupante', 'cabeza_con_casco', 'cabeza_sin_casco'],
              'training_ready': False, 'test_used': False,
              'limitation': 'Group and exact hash separation verified; original scene independence is not certified.'}
    out = Path('reports/separate-v2'); out.mkdir(parents=True, exist_ok=True)
    (out/'batch-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
