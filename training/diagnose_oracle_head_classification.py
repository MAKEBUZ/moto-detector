"""Oracle localization diagnostic; not deployable inference or accuracy estimate."""
import json,hashlib
from pathlib import Path
from collections import OrderedDict
import torch
from deterministic_gpu import configure_deterministic_gpu
from data import SeparateHelmetDataset
from models import build_detector
from train import fingerprint

@torch.inference_mode()
def main():
 configure_deterministic_gpu();torch.set_num_threads(4)
 base=Path('reports/training/reviewed-negative-reserve-01');out=base/'oracle-heads';out.mkdir(exist_ok=False);protocol=json.loads((base/'protocol.json').read_text());results=[]
 for kind in ('control','variant'):
  path=Path('runs')/protocol[kind]/'last.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,map_location='cpu',weights_only=True);c=ck['config'];root=Path(c['dataset'])
  for n,h in json.loads((root/'dataset.lock.json').read_text()).items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
  ds=SeparateHelmetDataset(root,'val',augment=False);model=build_detector(5,c['feature_stride'],c['proposal_multiplier'],deterministic_roi=True).cuda().eval();model.load_state_dict(ck['model']);initial=fingerprint(model);rows=[]
  for start in range(0,len(ds),2):
   batch=[ds[i] for i in range(start,min(start+2,len(ds)))];xs=[x.cuda() for x,t in batch];ts=[{k:v.cuda() for k,v in t.items()} for x,t in batch];ims,gt=model.transform(xs,ts);features=OrderedDict([('0',model.backbone(ims.tensors))]);boxes=[t['boxes'][t['labels']>=3] for t in gt]
   logits,_=model.roi_heads.box_predictor(model.roi_heads.box_head(model.roi_heads.box_roi_pool(features,boxes,ims.image_sizes)));scores=logits.softmax(-1).split([len(b) for b in boxes])
   for i,(t,ps) in enumerate(zip(gt,scores)):
    for j,prob in zip(torch.where(t['labels']>=3)[0].tolist(),ps):rows.append({'image':ds.rows[start+i]['image_id'],'gt_index':j,'label':int(t['labels'][j]),'box':t['boxes'][j].tolist(),'scores':prob.tolist(),'winner':int(prob.argmax()),'head_only_winner':int(prob[3:5].argmax())+3})
  assert fingerprint(model)==initial and hashlib.sha256(path.read_bytes()).hexdigest()==digest
  results.append({'kind':kind,'checkpoint_sha256':digest,'weights_buffers_unchanged':True,'heads':rows});del model,ck
 for a,b in zip(results[0]['heads'],results[1]['heads']):assert (a['image'],a['gt_index'],a['box'])==(b['image'],b['gt_index'],b['box'])
 (out/'diagnosis.json').write_text(json.dumps({'oracle_boxes':True,'training_performed':False,'test_used':False,'results':results},indent=2))
 for r in results:
  print(r['kind'])
  for h in r['heads']:print(h['image'],h['gt_index'],h['label'],h['winner'],h['head_only_winner'],[round(v,3) for v in h['scores']])
if __name__=='__main__':main()
