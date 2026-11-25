"""Frozen pre-NMS budget comparison; other thresholds and post-NMS cap fixed."""
import json,hashlib
from pathlib import Path
from collections import OrderedDict
import torch
from torchvision.ops import box_iou
from deterministic_gpu import configure_deterministic_gpu
from data import SeparateHelmetDataset
from models import build_detector
from metrics import detection_ap50
from evaluate_reserved import counts_at_threshold
from train import fingerprint

@torch.inference_mode()
def main():
 configure_deterministic_gpu();torch.set_num_threads(4)
 base=Path('reports/training/reviewed-negative-reserve-01');out=base/'budget-comparison';out.mkdir(exist_ok=False)
 protocol=json.loads((base/'protocol.json').read_text());root=Path('data/prepared/helmet-separate-v2-expanded-04')
 for name,h in json.loads((root/'dataset.lock.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
 ds=SeparateHelmetDataset(root,'val',augment=False);samples=[ds[i] for i in range(len(ds))];targets=[t for _,t in samples];records=[]
 for kind in ('control','variant'):
  path=Path('runs')/protocol[kind]/'last.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,map_location='cpu',weights_only=True);c=ck['config']
  model=build_detector(5,c['feature_stride'],c['proposal_multiplier'],deterministic_roi=True).cuda().eval();model.load_state_dict(ck['model']);initial=fingerprint(model)
  assert model.rpn.pre_nms_top_n()==2000 and model.rpn.post_nms_top_n()==800
  for budget in (2000,4000):
   model.rpn._pre_nms_top_n['testing']=budget;preds=[];heads=[]
   for start in range(0,len(samples),2):
    batch=samples[start:start+2];xs=[x.cuda() for x,t in batch];ts=[{k:v.cuda() for k,v in t.items()} for x,t in batch];ps=model(xs);preds.extend([{k:v.cpu() for k,v in p.items()} for p in ps]);ims,gt=model.transform(xs,ts);features=OrderedDict([('0',model.backbone(ims.tensors))]);proposals,_=model.rpn(ims,features)
    for i,(p,t) in enumerate(zip(ps,[t for x,t in batch])):
     for j in torch.where(t['labels']==4)[0].tolist():
      final_iou=box_iou(t['boxes'][j:j+1].cuda(),p['boxes'])[0];heads.append({'image':ds.rows[start+i]['image_id'],'gt_index':j,'rpn_max_iou':float(box_iou(gt[i]['boxes'][j:j+1],proposals[i]).max()),'covered05':bool(((final_iou>=.5)&(p['labels']==4)&(p['scores']>=.05)).any()),'covered50':bool(((final_iou>=.5)&(p['labels']==4)&(p['scores']>=.5)).any())})
   metrics=detection_ap50(preds,targets,(1,2,3,4))
   if budget==2000:
    assert abs(metrics['map50']-ck['validation']['map50'])<1e-6
    old=json.loads((base/(kind+'-predictions.json')).read_text());assert old==[{k:v.tolist() for k,v in p.items()} for p in preds]
   records.append({'kind':kind,'pre_nms':budget,'post_nms':800,'checkpoint_sha256':digest,'metrics':metrics,'counts05':counts_at_threshold(preds,targets,.05),'counts50':counts_at_threshold(preds,targets,.5),'heads':heads})
   (out/f'{kind}-{budget}-predictions.json').write_text(json.dumps([{'image':ds.rows[i]['image_id'],'prediction':{k:v.tolist() for k,v in p.items()},'target':{k:v.tolist() for k,v in targets[i].items()}} for i,p in enumerate(preds)],indent=2))
   assert fingerprint(model)==initial and hashlib.sha256(path.read_bytes()).hexdigest()==digest
  del model,ck
 result={'training_performed':False,'test_used':False,'weights_buffers_unchanged':True,'only_change':'RPN testing pre-NMS top-k 2000 vs 4000; post-NMS 800 unchanged','batch_size':2,'records':records}
 (out/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps([{k:r[k] for k in ('kind','pre_nms','counts05','counts50','heads')} for r in records],indent=2))
if __name__=='__main__':main()
