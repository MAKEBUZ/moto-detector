"""Frozen inference policies: head argmax and predicted-person geometry."""
import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
import torch
from torchvision.ops import clip_boxes_to_image, remove_small_boxes, batched_nms
from data import SeparateHelmetDataset
from models import build_detector, forbid_external_weights
from metrics import detection_ap50
from evaluate_reserved import counts_at_threshold
from train import fingerprint


def postprocess(boxes, probabilities, shape, roi):
    boxes=clip_boxes_to_image(boxes,shape)
    labels=torch.arange(probabilities.shape[1],device=boxes.device)[None].expand_as(probabilities)
    ids=torch.arange(len(boxes),device=boxes.device)[:,None].expand_as(probabilities)
    boxes=boxes[:,1:].reshape(-1,4);scores=probabilities[:,1:].reshape(-1)
    labels=labels[:,1:].reshape(-1);ids=ids[:,1:].reshape(-1)
    keep=torch.where(scores>roi.score_thresh)[0]
    boxes,scores,labels,ids=boxes[keep],scores[keep],labels[keep],ids[keep]
    keep=remove_small_boxes(boxes,min_size=1e-2)
    boxes,scores,labels,ids=boxes[keep],scores[keep],labels[keep],ids[keep]
    keep=batched_nms(boxes,scores,labels,roi.nms_thresh)[:roi.detections_per_img]
    return {'boxes':boxes[keep],'scores':scores[keep],'labels':labels[keep]},ids[keep]


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--budget',type=int,choices=(2000,4000),default=2000);args=parser.parse_args()
    out=Path('reports/training/object-reserve-01')/f'inference-policies-{args.budget}'
    out.mkdir(exist_ok=False)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    paired=read(Path('reports/training/object-reserve-01/comparison.json'))['records']
    torch.set_num_threads(4);forbid_external_weights();device='cuda' if torch.cuda.is_available() else 'cpu'
    (out/'protocol.json').write_text(json.dumps({'policies':['baseline','argmax','person_top40_conf50','combined'],'rpn_pre_nms':args.budget,'training':False,'test':False,'thresholds':[.05,.5],'selection':'all eight models; fixed rules, no parameter sweep'},indent=2),encoding='utf-8')
    results=[]
    for pair in paired:
        for model_type,run in [('control',pair['control_run']),('reserve',pair['run'])]:
            path=Path('runs')/run/'last.pt';checkpoint_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            ck=torch.load(path,map_location='cpu',weights_only=True);c=ck['config']
            assert ck['optimizer_updates']==1950 and not c['pretrained'] and c['initialization']=='random'
            root=Path(c['dataset'])
            for name,digest in read(root/'dataset.lock.json').items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
            ds=SeparateHelmetDataset(root,'val',augment=False)
            model=build_detector(5,c['feature_stride'],c['proposal_multiplier']).to(device)
            model.load_state_dict(ck['model']);model.eval();initial=fingerprint(model)
            model.rpn._pre_nms_top_n['testing']=args.budget
            preds={v:[] for v in ('regressed','argmax','person','combined')};targets=[];details=[]
            for start in range(0,len(ds),2):
                samples=[ds[i] for i in range(start,min(start+2,len(ds)))]
                xs=[x.to(device) for x,_ in samples];target=[t for _,t in samples]
                normal=model(xs)
                ims,_=model.transform(xs)
                features=OrderedDict([('0',model.backbone(ims.tensors))])
                proposals,_=model.rpn(ims,features)
                roi=model.roi_heads
                pooled=roi.box_roi_pool(features,proposals,ims.image_sizes)
                logits,deltas=roi.box_predictor(roi.box_head(pooled))
                probabilities=logits.softmax(-1).split([len(p) for p in proposals])
                decoded=roi.box_coder.decode(deltas,proposals).split([len(p) for p in proposals])
                official=roi.postprocess_detections(logits,deltas,proposals,ims.image_sizes)
                targets.extend(target)
                for i,(boxes,probs,shape) in enumerate(zip(decoded,probabilities,ims.image_sizes)):
                    unchanged_scores=probs.clone()
                    base,_=postprocess(boxes,probs,shape,roi)
                    persons=base['boxes'][(base['labels']==2)&(base['scores']>=.5)]
                    head_boxes=clip_boxes_to_image(boxes,shape)
                    geometry_ok=torch.zeros_like(probs,dtype=torch.bool)
                    for cls in (3,4):
                        centers=(head_boxes[:,cls,:2]+head_boxes[:,cls,2:])/2
                        if len(persons):
                            inside=(centers[:,None,0]>=persons[None,:,0])&(centers[:,None,0]<=persons[None,:,2])
                            inside&=(centers[:,None,1]>=persons[None,:,1])&(centers[:,None,1]<=persons[None,:,1]+.4*(persons[None,:,3]-persons[None,:,1]))
                            geometry_ok[:,cls]=inside.any(1)
                    winner=probs.argmax(1)
                    record={'image_id':ds.rows[start+i]['image_id'],
                            'target':{k:v.tolist() for k,v in target[i].items()},
                            'candidate_score_sha256':hashlib.sha256(probs.cpu().numpy().tobytes()).hexdigest()}
                    candidate_ids={}
                    for variant in preds:
                        filtered=probs.clone()
                        for cls in (3,4):
                            keep=torch.ones(len(probs),device=device,dtype=torch.bool)
                            if variant in ('argmax','combined'):keep&=winner==cls
                            if variant in ('person','combined'):keep&=geometry_ok[:,cls]
                            filtered[~keep,cls]=0
                        p,ids=postprocess(boxes,filtered,shape,roi)
                        if variant=='regressed':
                            assert all(torch.equal(p[k],official[j][i]) for j,k in enumerate(('boxes','scores','labels')))
                        assert torch.equal(p['scores'],probs[ids,p['labels']])
                        p=model.transform.postprocess([p],[shape],[xs[i].shape[-2:]])[0]
                        if variant=='regressed':assert all(torch.equal(p[k],normal[i][k]) for k in p)
                        cpu={k:v.cpu() for k,v in p.items()};preds[variant].append(cpu)
                        record[variant]={k:v.tolist() for k,v in cpu.items()}
                        record[variant]['proposal_indices']=ids.cpu().tolist()
                        candidate_ids[variant]=set(zip(ids.cpu().tolist(),p['labels'].cpu().tolist()))
                    record['surviving_candidate_sets_equal']=candidate_ids['regressed']==candidate_ids['combined']
                    assert torch.equal(probs,unchanged_scores)
                    details.append(record)
            metrics={v:detection_ap50(ps,targets,(1,2,3,4)) for v,ps in preds.items()}
            expected=json.loads((Path('reports/training/object-reserve-01/topk-2000-vs-4000')/(run+'.json')).read_text())['summary']['variants'][str(args.budget)]['metrics']['map50']
            assert abs(metrics['regressed']['map50']-expected)<1e-6
            assert fingerprint(model)==initial and hashlib.sha256(path.read_bytes()).hexdigest()==checkpoint_hash
            result={'run':run,'model_type':model_type,'images':pair['images'],'seed':pair['seed'],
                    'checkpoint_sha256':checkpoint_hash,'weights_and_buffers_unchanged':True,
                    'metrics':metrics,'counts':{v:{str(th):counts_at_threshold(ps,targets,th) for th in (.05,.5)} for v,ps in preds.items()},
                    'nohelmet_ap_delta_points':100*(metrics['combined']['per_class'][3]['ap50']-metrics['regressed']['per_class'][3]['ap50']),
                    'images_with_changed_survivors':sum(not d['surviving_candidate_sets_equal'] for d in details)}
            results.append(result)
            (out/(run+'.json')).write_text(json.dumps({'summary':result,'predictions':details},indent=2),encoding='utf-8')
            print(json.dumps({'run':run,'nohelmet_ap_delta_points':result['nohelmet_ap_delta_points'],'counts50':{v:result['counts'][v]['0.5'][3] for v in preds}}),flush=True)
            del model,ck
    (out/'comparison.json').write_text(json.dumps({'results':results,
        'intervention':'Head-only candidate gates before normal NMS: argmax across all 5 classes; center inside top40% of predicted occupant with score >=.5; combined. RPN budget set by CLI; otherwise unchanged. No GT used by gates.',
        'score_caveat':'Rejected head scores zeroed for filtering, retained scores and boxes unchanged. NMS/max100 survivors may change.',
        'method':'Frozen eval mode, batch2 original validation order; matching IoU .5. Both 0.05 and 0.5 thresholds fixed beforehand. Standard branch exactly matches torchvision and full model outputs.',
        'test_used':False,'training_performed':False,'deployment_changed':False},indent=2),encoding='utf-8')


if __name__=='__main__':main()
