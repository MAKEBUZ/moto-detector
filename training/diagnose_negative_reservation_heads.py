"""Frozen, batch-matched proposal/classification diagnosis on original validation."""
import json,hashlib
from pathlib import Path
from collections import OrderedDict
import torch
from torchvision.ops import box_iou,clip_boxes_to_image
from deterministic_gpu import configure_deterministic_gpu
from data import SeparateHelmetDataset
from models import build_detector
from train import fingerprint
from compare_head_box_regression import postprocess
from metrics import detection_ap50

@torch.inference_mode()
def main():
 configure_deterministic_gpu();torch.set_num_threads(4)
 base=Path('reports/training/reviewed-negative-reserve-01');out=base/'head-diagnosis';out.mkdir(exist_ok=False)
 protocol=json.loads((base/'protocol.json').read_text());results=[]
 for kind in ('control','variant'):
  path=Path('runs')/protocol[kind]/'last.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,map_location='cpu',weights_only=True);c=ck['config']
  root=Path(c['dataset'])
  for n,h in json.loads((root/'dataset.lock.json').read_text()).items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
  ds=SeparateHelmetDataset(root,'val',augment=False);model=build_detector(5,c['feature_stride'],c['proposal_multiplier'],deterministic_roi=True).cuda().eval();model.load_state_dict(ck['model']);initial=fingerprint(model);rows=[];preds=[];targets=[]
  for start in range(0,len(ds),2):
   samples=[ds[i] for i in range(start,min(start+2,len(ds)))];xs=[x.cuda() for x,t in samples];ts=[{k:v.cuda() for k,v in t.items()} for x,t in samples]
   normal=model(xs);ims,transformed=model.transform(xs,ts);features=OrderedDict([('0',model.backbone(ims.tensors))]);proposals,_=model.rpn(ims,features);roi=model.roi_heads
   logits,deltas=roi.box_predictor(roi.box_head(roi.box_roi_pool(features,proposals,ims.image_sizes)));sizes=[len(p) for p in proposals];probabilities=logits.softmax(-1).split(sizes);decoded=roi.box_coder.decode(deltas,proposals).split(sizes)
   for i,(p,probs,boxes,gt,shape) in enumerate(zip(proposals,probabilities,decoded,transformed,ims.image_sizes)):
    final,ids=postprocess(boxes,probs,shape,roi);restored=model.transform.postprocess([{k:v.clone() for k,v in final.items()}],[shape],[xs[i].shape[-2:]])[0]
    assert all(torch.equal(restored[k],normal[i][k]) for k in restored)
    preds.append({k:v.cpu() for k,v in normal[i].items()});targets.append(samples[i][1]);adjusted=clip_boxes_to_image(boxes,shape)
    for j in torch.where(gt['labels']==4)[0].tolist():
     box=gt['boxes'][j:j+1];overlap=box_iou(box,p)[0];after=box_iou(box,adjusted[:,4,:])[0];near=overlap>=.5;best=int(overlap.argmax());eligible=(after>=.5)&(probs[:,4]>.05)
     detected=bool(((box_iou(box,final['boxes'])[0]>=.5)&(final['labels']==4)).any())
     rows.append({'image':ds.rows[start+i]['image_id'],'gt_index':j,'rpn_max_iou':float(overlap.max()),'near_count':int(near.sum()),'near_winner_counts':torch.bincount(probs[near].argmax(-1),minlength=5).tolist(),'near_max_nohelmet_score':float(probs[near,4].max()) if near.any() else None,'best_proposal_scores':probs[best].tolist(),'best_proposal_adjusted_iou':float(after[best]),'near_scored05':int((near&(probs[:,4]>.05)).sum()),'near_scored05_after_iou50':int((near&eligible).sum()),'all_adjusted_scored05':int(eligible.sum()),'detected05':detected,'candidates':[{'proposal_id':int(k),'iou_before':float(overlap[k]),'iou_after':float(after[k]),'scores':probs[k].tolist(),'survives_nohelmet':bool(((ids==k)&(final['labels']==4)).any())} for k in torch.where(near|eligible)[0].tolist()]})
  metrics=detection_ap50(preds,targets,(1,2,3,4));assert abs(metrics['map50']-ck['validation']['map50'])<1e-6
  assert fingerprint(model)==initial and hashlib.sha256(path.read_bytes()).hexdigest()==digest
  results.append({'kind':kind,'checkpoint_sha256':digest,'weights_buffers_unchanged':True,'metrics':metrics,'heads':rows});del model,ck
 (out/'diagnosis.json').write_text(json.dumps({'training_performed':False,'test_used':False,'batch_size':2,'results':results},indent=2))
 for a,b in zip(results[0]['heads'],results[1]['heads']):
  assert (a['image'],a['gt_index'])==(b['image'],b['gt_index'])
  print(json.dumps([{k:v for k,v in x.items() if k!='candidates'} for x in (a,b)]))
if __name__=='__main__':main()
