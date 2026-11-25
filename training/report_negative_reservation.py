"""Compare deterministic original versus reviewed negative reservation."""
import json,hashlib
from pathlib import Path
import torch
from deterministic_gpu import configure_deterministic_gpu
from data import SeparateHelmetDataset
from models import build_detector
from metrics import detection_ap50
from evaluate_reserved import counts_at_threshold

@torch.inference_mode()
def main():
 configure_deterministic_gpu();torch.set_num_threads(4)
 out=Path('reports/training/reviewed-negative-reserve-01')
 protocol=json.loads((out/'protocol.json').read_text())
 root=Path('data/prepared/helmet-separate-v2-expanded-04')
 assert hashlib.sha256((root/'dataset.lock.json').read_bytes()).hexdigest()==protocol['dataset_lock_sha256']
 for name,h in json.loads((root/'dataset.lock.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
 ds=SeparateHelmetDataset(root,'val',augment=False);samples=[ds[i] for i in range(len(ds))];targets=[t for _,t in samples]
 records=[];configs=[]
 for kind in ('control','variant'):
  folder=Path('runs')/protocol[kind];ck=torch.load(folder/'last.pt',map_location='cpu',weights_only=True);c=ck['config'];configs.append(c)
  assert ck['optimizer_updates']==1950 and ck['image_presentations']==3900 and not c['pretrained'] and c['deterministic_gpu']
  model=build_detector(5,c['feature_stride'],c['proposal_multiplier'],deterministic_roi=True).cuda().eval();model.load_state_dict(ck['model']);preds=[]
  for start in range(0,len(samples),2):preds.extend([{k:v.cpu() for k,v in p.items()} for p in model([x.cuda() for x,_ in samples[start:start+2]])])
  metrics=detection_ap50(preds,targets,(1,2,3,4));assert abs(metrics['map50']-ck['validation']['map50'])<1e-6
  exposure=[json.loads(l) for l in (folder/'negative-exposure.jsonl').read_text().splitlines()]
  stats={stage:{'observations':sum(r['stage']==stage for r in exposure),'selected_near':sum(r['stage']==stage and r['thresholds']['0.5']['selected']['0']>0 for r in exposure),'swaps':sum(r.get('reserved_negative_changes',0) for r in exposure if r['stage']==stage)} for stage in ('rpn','roi')}
  records.append({'kind':kind,'metrics':metrics,'counts05':counts_at_threshold(preds,targets,.05),'counts50':counts_at_threshold(preds,targets,.5),'exposure':stats})
  (out/(kind+'-predictions.json')).write_text(json.dumps([{k:v.tolist() for k,v in p.items()} for p in preds]))
  del model,ck
 for key in ('initial_state_sha256','seed','batch_size','workers','learning_rate','photometric_policy','feature_stride','proposal_multiplier','gpu_reproducibility_settings'):assert configs[0][key]==configs[1][key],key
 (out/'comparison.json').write_text(json.dumps(records,indent=2))
 print(json.dumps(records,indent=2))
if __name__=='__main__':main()
