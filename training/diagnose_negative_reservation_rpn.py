"""Trace frozen RPN filtering with original batch geometry and exact output checks."""
import json,hashlib
from pathlib import Path
from collections import OrderedDict
import torch
from torchvision.ops import box_iou,clip_boxes_to_image,remove_small_boxes,batched_nms
from torchvision.models.detection.rpn import concat_box_prediction_layers
from deterministic_gpu import configure_deterministic_gpu
from data import SeparateHelmetDataset
from models import build_detector
from train import fingerprint

@torch.inference_mode()
def main():
 configure_deterministic_gpu();torch.set_num_threads(4)
 base=Path('reports/training/reviewed-negative-reserve-01');out=base/'rpn-diagnosis';out.mkdir(exist_ok=False)
 protocol=json.loads((base/'protocol.json').read_text());results=[]
 previous=json.loads((base/'head-diagnosis/diagnosis.json').read_text())['results']
 for kind in ('control','variant'):
  path=Path('runs')/protocol[kind]/'last.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest();ck=torch.load(path,map_location='cpu',weights_only=True);c=ck['config'];root=Path(c['dataset'])
  for n,h in json.loads((root/'dataset.lock.json').read_text()).items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
  ds=SeparateHelmetDataset(root,'val',augment=False);model=build_detector(5,c['feature_stride'],c['proposal_multiplier'],deterministic_roi=True).cuda().eval();model.load_state_dict(ck['model']);initial=fingerprint(model);rows=[];rpn=model.rpn
  for start in range(0,len(ds),2):
   samples=[ds[i] for i in range(start,min(start+2,len(ds)))];xs=[x.cuda() for x,t in samples];ts=[{k:v.cuda() for k,v in t.items()} for x,t in samples];ims,gt=model.transform(xs,ts)
   features=OrderedDict([('0',model.backbone(ims.tensors))]);official,_=rpn(ims,features);fs=list(features.values());obj,delta=rpn.head(fs);anchors=rpn.anchor_generator(ims,fs);counts=[o[0].numel() for o in obj];assert len(counts)==1
   obj,delta=concat_box_prediction_layers(obj,delta);decoded=rpn.box_coder.decode(delta.detach(),anchors).view(len(xs),-1,4);scores=obj.reshape(len(xs),-1);top=rpn._get_top_n_idx(scores,counts)
   for i,shape in enumerate(ims.image_sizes):
    raw=decoded[i];clipped=clip_boxes_to_image(raw,shape);prob=scores[i].sigmoid();ids=top[i];stages={'decoded_all':(raw,torch.arange(len(raw),device='cuda')),'topk_raw':(raw[ids],ids),'clipped':(clipped[ids],ids)}
    ids=ids[remove_small_boxes(clipped[ids],rpn.min_size)];stages['size_filter']=(clipped[ids],ids)
    ids=ids[prob[ids]>=rpn.score_thresh];stages['score_filter']=(clipped[ids],ids)
    ids=ids[batched_nms(clipped[ids],prob[ids],torch.zeros_like(ids),rpn.nms_thresh)];stages['nms']=(clipped[ids],ids)
    ids=ids[:rpn.post_nms_top_n()];stages['final']=(clipped[ids],ids);assert torch.equal(clipped[ids],official[i])
    for j in torch.where(gt[i]['labels']==4)[0].tolist():
     box=gt[i]['boxes'][j:j+1];aiou=box_iou(box,anchors[i])[0];diou=box_iou(box,raw)[0];ciou=box_iou(box,clipped)[0];near=torch.where(diou>=.5)[0]
     stats={name:{'total':len(b),'near50':int((box_iou(box,b)[0]>=.5).sum()),'max_iou':float(box_iou(box,b).max()) if len(b) else 0.} for name,(b,ix) in stages.items()}
     best=int(diou.argmax());rank=int((scores[i]>scores[i,best]).sum())+1
     rows.append({'image':ds.rows[start+i]['image_id'],'gt_index':j,'anchor_max_iou':float(aiou.max()),'anchor_near50':int((aiou>=.5).sum()),'near_anchors_still_near_after_regression':int(((aiou>=.5)&(diou>=.5)).sum()),'best_decoded_score':float(prob[best]),'best_decoded_rank':rank,'best_near_rank':min([int((scores[i]>scores[i,k]).sum())+1 for k in near]) if len(near) else None,'stages':stats,'near_candidates':[{'anchor_id':int(k),'anchor_iou':float(aiou[k]),'decoded_iou':float(diou[k]),'clipped_iou':float(ciou[k]),'score':float(prob[k]),'rank':int((scores[i]>scores[i,k]).sum())+1,'survives':{n:bool((ix==k).any()) for n,(b,ix) in stages.items()}} for k in near.tolist()]})
  old=next(r for r in previous if r['kind']==kind)['heads']
  assert len(rows)==len(old)==6
  for a,b in zip(rows,old):assert a['image']==b['image'] and abs(a['stages']['final']['max_iou']-b['rpn_max_iou'])<1e-6
  assert fingerprint(model)==initial and hashlib.sha256(path.read_bytes()).hexdigest()==digest
  results.append({'kind':kind,'checkpoint_sha256':digest,'weights_buffers_unchanged':True,'pre_nms':rpn.pre_nms_top_n(),'post_nms':rpn.post_nms_top_n(),'score_threshold':rpn.score_thresh,'nms_threshold':rpn.nms_thresh,'heads':rows});del model,ck
 (out/'diagnosis.json').write_text(json.dumps({'training_performed':False,'test_used':False,'batch_size':2,'results':results},indent=2))
 for r in results:
  for h in r['heads']:print(r['kind'],h['image'],h['gt_index'],'anchors',h['anchor_near50'],'stages',[(k,v['near50']) for k,v in h['stages'].items()],'best near rank',h['best_near_rank'])
if __name__=='__main__':main()
