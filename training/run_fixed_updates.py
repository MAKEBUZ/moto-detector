"""Execute the prespecified two-dataset, two-seed experiment from scratch."""
import hashlib
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rpn-object-reserve', action='store_true')
    args=parser.parse_args()
    out = Path('reports/training/object-reserve-01' if args.rpn_object_reserve else 'reports/training/fixed-updates-01')
    out.mkdir(parents=True, exist_ok=False)
    datasets = ['helmet-separate-v2-expanded-02', 'helmet-separate-v2-expanded-03']
    roots = [Path('data/prepared')/d for d in datasets]
    for n in ('val-manifest.jsonl','val-coco.json'):
        assert (roots[0]/n).read_bytes() == (roots[1]/n).read_bytes()
    jobs = []
    for seed in (20261124,20261125):
        for count,root in zip((35,44),roots):
            name = f'detector-{"reserve1950" if args.rpn_object_reserve else "fixed1950"}-n{count}-s{seed}'
            assert not (Path('runs')/name).exists()
            jobs.append({'run':name,'dataset':str(root),'seed':seed,'images':count})
    protocol = {'jobs':jobs,'optimizer_updates':1950,'checkpoint':'last.pt',
                'selection':'fixed final update, not best validation epoch',
                'confidence_threshold':0.5,'matching_iou':0.5,
                'primary_metric':'nohelmet AP50 at final update',
                'pretrained':False,'initialization':'random','test_used':False,
                'rpn_object_reserve':args.rpn_object_reserve,
                'reservation_policy':'Keep vanilla sampled counts and negative anchors. Replace redundant sampled positives to give one to each GT with assigned positives. Independent repair RNG seeded run_seed+1000000. Fail if budget cannot cover eligible GT.',
                'dataset_lock_sha256':{str(p):hashlib.sha256((p/'dataset.lock.json').read_bytes()).hexdigest() for p in roots},
                'limitations':['Two seeds are exploratory, not a statistical stability guarantee.',
                               'Same updates, but 35-image loader has one single-image batch per epoch; image presentations differ.',
                               'Validation has five repeatedly inspected images; test is not used.']}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for job in jobs:
        cmd = [str(Path('.venv/Scripts/python.exe').resolve()), 'training/train.py',
               '--task','detector','--separate-classes','--dataset',job['dataset'],
               '--epochs','150','--batch-size','2','--workers','0','--seed',str(job['seed']),
               '--constant-lr','--max-updates','1950','--head-probe-every','10',
               '--photometric','--feature-stride','8','--proposal-multiplier','4','--run',job['run']]
        if args.rpn_object_reserve: cmd.append('--rpn-object-reserve')
        print('START '+job['run'],flush=True)
        with (out/(job['run']+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((Path('runs')/job['run']/'result.json').read_text())
        assert result['optimizer_updates']==1950
        print('DONE '+job['run']+' '+str(result['final_validation']),flush=True)


if __name__=='__main__': main()
