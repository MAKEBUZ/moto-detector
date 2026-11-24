"""Trace frozen RPN candidates by anchor identity through classification and regression."""
import hashlib
import json
from collections import Counter, OrderedDict
from pathlib import Path
import torch
from torchvision.ops import box_iou, clip_boxes_to_image, remove_small_boxes, batched_nms
from torchvision.models.detection.rpn import concat_box_prediction_layers
from compare_head_box_regression import postprocess
from data import SeparateHelmetDataset
from models import build_detector, forbid_external_weights
from train import fingerprint


def matched_status(pred, target, threshold):
    used=set(); result={}
    for i in torch.argsort(pred['scores'],descending=True).tolist():
        if int(pred['labels'][i])!=4 or float(pred['scores'][i])<threshold:continue
        overlaps=box_iou(pred['boxes'][i:i+1],target['boxes'])[0]
        choices=[j for j in range(len(overlaps)) if int(target['labels'][j])==4 and j not in used and overlaps[j]>=.5]
        if choices:
            used.add(max(choices,key=lambda j:float(overlaps[j])));result[i]='tp'
        elif any(overlaps[j]>=.5 and int(target['labels'][j])==4 for j in range(len(overlaps))):result[i]='duplicate'
        elif any(overlaps[j]>=.5 and int(target['labels'][j])==3 for j in range(len(overlaps))):result[i]='helmet_confusion'
        elif any(overlaps[j]>=.1 and int(target['labels'][j])==4 for j in range(len(overlaps))):result[i]='nohelmet_localization'
        else:result[i]='other_region'
    return result


@torch.inference_mode()
def main():
    out=Path('reports/training/object-reserve-01/additional-proposals');out.mkdir(exist_ok=False)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    old=Path('reports/training/object-reserve-01/topk-2000-vs-4000')
    jobs=read(old/'comparison.json')['results']
    protocol={'training_performed':False,'test_used':False,'budgets':[2000,4000],'post_nms':800,
              'matching':'Same decoded RPN anchor index, same image. Batch 2,2,1; frozen eval.',
              'fp_categories':'Exclusive: duplicate nohelmet IoU>=.5; helmet IoU>=.5; nohelmet .1<=IoU<.5; other_region. Other region is not certified background.',
              'regression':'Per-candidate maximum nohelmet IoU before/after; counterfactual replaces ONLY nohelmet boxes, retaining all candidate scores then reruns normal filtering/NMS/max100.',
              'thresholds':[.05,.5]}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.set_num_threads(4);forbid_external_weights();device='cuda' if torch.cuda.is_available() else 'cpu'
    summaries=[]
    for job in jobs:
        run=job['run'];path=Path('runs')/run/'last.pt';sha=hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha==job['checkpoint_sha256']
        ck=torch.load(path,map_location='cpu',weights_only=True);c=ck['config']
        assert not c['pretrained'] and c['initialization']=='random' and ck['optimizer_updates']==1950
        root=Path(c['dataset'])
        for name,h in read(root/'dataset.lock.json').items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
        ds=SeparateHelmetDataset(root,'val',augment=False)
        model=build_detector(5,c['feature_stride'],c['proposal_multiplier']).to(device);model.load_state_dict(ck['model']);model.eval();initial=fingerprint(model)
        reference=read(old/(run+'.json'))
        rows=[];totals={str(b):{str(t):Counter() for t in (.05,.5)} for b in (2000,4000)};ablation={str(t):Counter() for t in (.05,.5)}
        for start in range(0,len(ds),2):
            samples=[ds[i] for i in range(start,min(start+2,len(ds)))]
            xs=[x.to(device) for x,_ in samples];ts=[{k:v.to(device) for k,v in t.items()} for _,t in samples]
            ims,targets=model.transform(xs,ts);features=OrderedDict([('0',model.backbone(ims.tensors))]);rpn=model.rpn;roi=model.roi_heads
            obj,delta=rpn.head(list(features.values()));counts=[o[0].numel() for o in obj];assert len(counts)==1
            anchors=rpn.anchor_generator(ims,list(features.values()));obj,delta=concat_box_prediction_layers(obj,delta)
            decoded=rpn.box_coder.decode(delta,anchors).reshape(len(xs),-1,4);scores=obj.reshape(len(xs),-1).sigmoid()
            collected={}
            for budget in (2000,4000):
                rpn._pre_nms_top_n['testing']=budget
                official,_=rpn.filter_proposals(decoded,obj,ims.image_sizes,counts)
                props=[];identities=[]
                for i,shape in enumerate(ims.image_sizes):
                    ids=torch.topk(obj.reshape(len(xs),-1)[i],min(budget,decoded.shape[1])).indices
                    boxes=clip_boxes_to_image(decoded[i,ids],shape)
                    keep=remove_small_boxes(boxes,rpn.min_size);ids,boxes=ids[keep],boxes[keep]
                    keep=torch.where(scores[i,ids]>=rpn.score_thresh)[0];ids,boxes=ids[keep],boxes[keep]
                    keep=batched_nms(boxes,scores[i,ids],torch.zeros_like(ids),rpn.nms_thresh)[:800]
                    ids,boxes=ids[keep],boxes[keep];assert torch.equal(boxes,official[i]);props.append(boxes);identities.append(ids)
                logits,deltas=roi.box_predictor(roi.box_head(roi.box_roi_pool(features,props,ims.image_sizes)))
                probs=logits.softmax(-1).split([len(p) for p in props]);boxes=roi.box_coder.decode(deltas,props).split([len(p) for p in props])
                collected[budget]=[]
                for i,shape in enumerate(ims.image_sizes):
                    pred,indices=postprocess(boxes[i],probs[i],shape,roi)
                    scaled=model.transform.postprocess([{k:v.clone() for k,v in pred.items()}],[shape],[xs[i].shape[-2:]])[0]
                    expected=reference['predictions'][str(budget)][start+i]['prediction']
                    for k in scaled:assert torch.allclose(scaled[k].cpu(),torch.tensor(expected[k],dtype=scaled[k].dtype),atol=1e-5,rtol=1e-5)
                    collected[budget].append((props[i],identities[i],probs[i],boxes[i],pred,indices))
            for i,target in enumerate(targets):
                base=collected[2000][i];expanded=collected[4000][i];base_ids=set(base[1].tolist());new_ids=set(expanded[1].tolist())-base_ids
                # Common candidates retain geometry and scores (allow GPU batching roundoff).
                lookup={a:j for j,a in enumerate(base[1].tolist())}
                for j,a in enumerate(expanded[1].tolist()):
                    if a in lookup:
                        q=lookup[a];assert torch.equal(expanded[0][j],base[0][q]);assert torch.allclose(expanded[2][j],base[2][q],atol=1e-5,rtol=1e-5)
                details=[]
                for budget,entry in ((2000,base),(4000,expanded)):
                    proposal,ids,prob,adjusted,pred,index=entry
                    statuses={str(th):matched_status(pred,target,th) for th in (.05,.5)}
                    for th,states in statuses.items():
                        for n,status in states.items():
                            origin='additional' if int(ids[index[n]]) in new_ids else 'shared_or_baseline'
                            totals[str(budget)][th][origin+':'+status]+=1
                    nh=target['boxes'][target['labels']==4];helmet=target['boxes'][target['labels']==3]
                    for n in statuses['0.05']:
                        j=int(index[n]);pre=float(box_iou(proposal[j:j+1],nh).max()) if len(nh) else 0.;post=float(box_iou(pred['boxes'][n:n+1],nh).max()) if len(nh) else 0.
                        details.append({'budget':budget,'anchor_id':int(ids[j]),'additional':int(ids[j]) in new_ids,'score':float(pred['scores'][n]),
                            'probabilities':prob[j].tolist(),'argmax':int(prob[j].argmax()),'proposal_box':proposal[j].tolist(),'adjusted_box':pred['boxes'][n].tolist(),
                            'nohelmet_iou_before':pre,'nohelmet_iou_after':post,'delta_iou':post-pre,
                            'helmet_iou_before':float(box_iou(proposal[j:j+1],helmet).max()) if len(helmet) else 0.,
                            'status05':statuses['0.05'][n],'status50':statuses['0.5'].get(n)})
                proposal,ids,prob,adjusted,pred,index=expanded
                replacement=adjusted.clone();replacement[:,4,:]=proposal
                counter,_=postprocess(replacement,prob,ims.image_sizes[i],roi)
                for th in (.05,.5):ablation[str(th)].update(matched_status(counter,target,th).values())
                new_positions=[j for j,a in enumerate(ids.tolist()) if a in new_ids]
                rows.append({'image_id':ds.rows[start+i]['image_id'],'additional_proposals':len(new_ids),'lost_baseline_proposals':len(base_ids-set(ids.tolist())),
                             'additional_argmax_counts':dict(Counter(int(prob[j].argmax()) for j in new_positions)),
                             'additional_nohelmet_score_above05':sum(float(prob[j,4])>.05 for j in new_positions),'detections':details})
        model.rpn._pre_nms_top_n['testing']=2000
        assert fingerprint(model)==initial and hashlib.sha256(path.read_bytes()).hexdigest()==sha
        summary={'run':run,'images':job['images'],'seed':job['seed'],'variant':job['variant'],'checkpoint_sha256':sha,'weights_unchanged':True,
                 'counts':totals,'nohelmet_unregressed_4000':ablation,'additional_proposals':sum(r['additional_proposals'] for r in rows),
                 'lost_baseline_proposals':sum(r['lost_baseline_proposals'] for r in rows)}
        summaries.append(summary);(out/(run+'.json')).write_text(json.dumps({'summary':summary,'images':rows},indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)
        del model,ck
    (out/'comparison.json').write_text(json.dumps({'results':summaries},indent=2),encoding='utf-8')


if __name__=='__main__':main()
