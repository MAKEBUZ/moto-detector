"""Frozen RPN ranking, supervision assignment and competition diagnostics."""
import hashlib
import json
import statistics
from collections import OrderedDict, Counter
from pathlib import Path
import torch
from torchvision.ops import box_iou
from data import SeparateHelmetDataset
from models import build_detector, forbid_external_weights
from audit_positive_sampling import proposals, max_iou
from train import fingerprint


@torch.inference_mode()
def main():
    out=Path('reports/training/object-reserve-01/rpn-ranking')
    out.mkdir(exist_ok=False)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    pairs=read(Path('reports/training/object-reserve-01/comparison.json'))['records']
    torch.set_num_threads(4);forbid_external_weights();device='cuda' if torch.cuda.is_available() else 'cpu'
    reports=[]
    for pair in pairs:
        for variant,run in [('control',pair['control_run']),('reserve',pair['run'])]:
            path=Path('runs')/run/'last.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest()
            ck=torch.load(path,map_location='cpu',weights_only=True);c=ck['config'];root=Path(c['dataset'])
            assert not c['pretrained'] and c['initialization']=='random' and ck['optimizer_updates']==1950
            for name,h in read(root/'dataset.lock.json').items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
            model=build_detector(5,c['feature_stride'],c['proposal_multiplier']).to(device)
            model.load_state_dict(ck['model']);model.eval();before=fingerprint(model)
            heads=[];competition=[]
            for split in ('train','val'):
                ds=SeparateHelmetDataset(root,split,augment=False)
                for start in range(0,len(ds),2):
                    samples=[ds[i] for i in range(start,min(start+2,len(ds)))]
                    ims,targets=model.transform([x.to(device) for x,_ in samples],[{k:v.to(device) for k,v in t.items()} for _,t in samples])
                    features=OrderedDict([('0',model.backbone(ims.tensors))])
                    anchors,final,stages,logits=proposals(model,ims,features)
                    for i,t in enumerate(targets):
                        meta=ds.rows[start+i];gt=t['boxes'];classes=t['labels'];a_iou=box_iou(gt,anchors[i])
                        assignment=model.rpn.proposal_matcher(a_iou)
                        d_iou=box_iou(gt,stages[i]['decoded_clipped'])
                        top_ids=logits[i].topk(min(2000,len(logits[i]))).indices
                        best_overlap,best_gt=d_iou[:,top_ids].max(0)
                        counts=Counter('low_overlap' if float(ov)<.5 else str(int(classes[j])) for ov,j in zip(best_overlap,best_gt))
                        competition.append({'split':split,'image_id':meta['image_id'],'pre_nms_top_count':len(top_ids),
                            'decoded_top_matches':dict(counts),
                            'definition':'Assign each decoded top-2000 box to highest-IoU GT if IoU>=0.5, else low_overlap. Multiple boxes may match one GT; low_overlap is not certified background.'})
                        cutoff=logits[i][top_ids[-1]]
                        for j,cls in enumerate(classes.tolist()):
                            if cls not in (3,4):continue
                            positive=assignment==j
                            positive_count=int(positive.sum())
                            good=d_iou[j]>=.5
                            candidate_ids=torch.where(good)[0]
                            k=int(candidate_ids[logits[i][candidate_ids].argmax()]) if len(candidate_ids) else None
                            rank=int((logits[i]>logits[i,k]).sum())+1 if k is not None else None
                            best_anchor=int(a_iou[j].argmax())
                            label_before=int(assignment[k]) if k is not None else None
                            assigned_class=int(classes[label_before]) if label_before is not None and label_before>=0 else None
                            heads.append({'split':split,'image_id':meta['image_id'],'entity_id':meta['annotations'][j]['entity_id'],'label':cls,
                                'min_side_model_px':float((gt[j,2:]-gt[j,:2]).min()),
                                'anchor_max_iou':float(a_iou[j].max()),'assigned_positive_anchors':positive_count,
                                'assigned_positive_mean_score':float(logits[i][positive].sigmoid().mean()) if positive_count else None,
                                'assigned_positive_top2000':int(positive[top_ids].sum()),
                                'best_anchor_score':float(logits[i,best_anchor].sigmoid()),
                                'best_anchor_decoded_iou':float(d_iou[j,best_anchor]),
                                'best_anchor_assignment':int(assignment[best_anchor]),
                                'decoded_cover_count':int(good.sum()),'decoded_max_iou':float(d_iou[j].max()),
                                'best_cover_rank':rank,'best_cover_score':float(logits[i,k].sigmoid()) if k is not None else None,
                                'cutoff_score':float(cutoff.sigmoid()),
                                'logit_gap_to_cutoff':float(logits[i,k]-cutoff) if k is not None else None,
                                'best_cover_origin_anchor_iou':float(a_iou[j,k]) if k is not None else None,
                                'best_cover_origin_anchor_assigned_class':assigned_class,
                                'best_cover_origin_anchor_assignment':label_before,
                                'topk_iou':float(max_iou(gt[j:j+1],stages[i]['topk_clipped'])[0]),
                                'final_iou':float(max_iou(gt[j:j+1],final[i])[0])})
            assert fingerprint(model)==before and hashlib.sha256(path.read_bytes()).hexdigest()==digest
            report={'run':run,'variant':variant,'images':pair['images'],'seed':pair['seed'],'checkpoint_sha256':digest,
                    'state_unchanged':True,'heads':heads,'competition':competition,'summary':{}}
            for split in ('train','val'):
                hs=[h for h in heads if h['split']==split and h['label']==4]
                report['summary'][split]={'heads':len(hs),'anchor_cover50':sum(h['anchor_max_iou']>=.5 for h in hs),
                    'no_assigned_positive':sum(h['assigned_positive_anchors']==0 for h in hs),
                    'median_positive_score':statistics.median(h['assigned_positive_mean_score'] for h in hs if h['assigned_positive_mean_score'] is not None),
                    'decoded_cover50':sum(h['decoded_max_iou']>=.5 for h in hs),
                    'hypothetical_pre_nms_coverage':{str(k):sum(h['best_cover_rank'] is not None and h['best_cover_rank']<=k for h in hs) for k in (2000,4000,8000)},
                    'final_cover50':sum(h['final_iou']>=.5 for h in hs)}
            (out/(run+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8');reports.append(report)
            print(json.dumps({'run':run,'summary':report['summary']}),flush=True)
            del model,ck
    (out/'summary.json').write_text(json.dumps({'reports':reports,
        'method':'Frozen eval mode train/val, unaugmented fixed order batch2. GT matching used only for diagnosis, not fed into RPN prediction. Positive assignment follows current training matcher including low-quality matching.',
        'limits':'Positive scores are uncalibrated. Supervision labels and train inference here are not historical augmented training samples. Hypothetical top-k coverage is before NMS/cap/classification, not detector recall or a deployed parameter change. Top boxes may be redundant and low-overlap does not prove background.',
        'training_performed':False,'test_used':False},indent=2),encoding='utf-8')


if __name__=='__main__':main()
