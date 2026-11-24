"""Evaluate all prespecified final checkpoints on unchanged validation only."""
import hashlib
import json
import shutil
from pathlib import Path
import torch
from data import SeparateHelmetDataset
from models import build_detector, forbid_external_weights
from metrics import detection_ap50
from evaluate_reserved import counts_at_threshold


@torch.inference_mode()
def main():
    out = Path('reports/training/negative-exposure-02')
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    protocol = read(out/'protocol.json')
    torch.set_num_threads(4); forbid_external_weights()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    protocol['jobs']=[j for seed in (20261124,20261125) for j in protocol['controls_reused']+protocol['jobs'] if j['seed']==seed]
    records = []; configs = []; initial_by_seed = {}
    for job in protocol['jobs']:
        run = Path('runs')/job['run']; root = Path(job['dataset'])
        assert hashlib.sha256((root/'dataset.lock.json').read_bytes()).hexdigest()==protocol['dataset_lock_sha256'][str(root)]
        for n,digest in read(root/'dataset.lock.json').items():
            assert hashlib.sha256((root/n).read_bytes()).hexdigest()==digest
        checkpoint = run/protocol['checkpoint']
        ck = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if 'checkpoint_sha256' in job: assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==job['checkpoint_sha256']
        c = ck['config'];configs.append(c)
        assert not c.get('rpn_object_reserve',False)
        assert ck['image_presentations']==3900
        assert c['pretrained'] is False and c['initialization']=='random'
        assert ck['optimizer_updates']==protocol['optimizer_updates']==1950
        assert c['seed']==job['seed'] and c['train_examples']==job['images']
        if job['seed'] in initial_by_seed: assert initial_by_seed[job['seed']]==c['initial_state_sha256']
        initial_by_seed[job['seed']]=c['initial_state_sha256']
        steps={int(state['step']) for state in ck['optimizer']['state'].values()}
        assert steps=={1950},steps
        ds = SeparateHelmetDataset(root,'val',augment=False)
        samples=[ds[i] for i in range(len(ds))];targets=[t for _,t in samples]
        model=build_detector(5,c['feature_stride'],c['proposal_multiplier'])
        model.load_state_dict(ck['model']);model.to(device).eval();preds=[]
        for start in range(0,len(samples),2):
            preds.extend([{k:v.cpu() for k,v in p.items()} for p in model([x.to(device) for x,_ in samples[start:start+2]])])
        metrics=detection_ap50(preds,targets,(1,2,3,4))
        assert abs(metrics['map50']-ck['validation']['map50'])<1e-6
        row={**job,'checkpoint':protocol['checkpoint'],'epoch':ck['epoch'],
             'optimizer_updates':ck['optimizer_updates'],'image_presentations':ck['image_presentations'],
             'initial_state_sha256':c['initial_state_sha256'],
             'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             'metrics':metrics,'counts50':counts_at_threshold(preds,targets,.5),
             'counts05':counts_at_threshold(preds,targets,.05)}
        records.append(row)
        dest=out/job['run'];dest.mkdir(exist_ok=False)
        for n in ('config.json','history.json','result.json','requirements.lock.txt'):
            shutil.copyfile(run/n,dest/n)
        shutil.copytree(run/'head-probes',dest/'head-probes')
        (dest/'predictions.json').write_text(json.dumps([{'image_id':r['image_id'],'prediction':{k:v.tolist() for k,v in p.items()},'target':{k:v.tolist() for k,v in t.items()}} for r,p,t in zip(ds.rows,preds,targets)],indent=2),encoding='utf-8')
        del model,ck
    fixed=['batch_size','workers','constant_lr','learning_rate','photometric_policy','feature_stride','proposal_multiplier','head_probe_every','classes','parameters','max_updates']
    for key in fixed: assert all(c[key]==configs[0][key] for c in configs),key
    assert len(set(initial_by_seed.values()))==2
    pairs=[]
    for seed in initial_by_seed:
        old,new=[r for r in records if r['seed']==seed]
        pairs.append({'seed':seed,'nohelmet_ap_delta_points':100*(new['metrics']['per_class'][3]['ap50']-old['metrics']['per_class'][3]['ap50']),
                      'nohelmet_tp50_delta':new['counts50'][3]['tp']-old['counts50'][3]['tp']})
    summary={'protocol_sha256':hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),'records':records,'paired_deltas':pairs,'fixed_settings_verified':fixed,'test_used':False,'checkpoint_selection':'last at update 1950'}
    (out/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'records':records,'paired_deltas':pairs},indent=2))

if __name__=='__main__':main()
