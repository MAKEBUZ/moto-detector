"""Paired frozen validation diagnosis of controls and object-reserved RPN runs."""
import hashlib
import argparse
import json
from collections import OrderedDict
from pathlib import Path
import torch
from torchvision.ops import box_iou, clip_boxes_to_image
from data import SeparateHelmetDataset
from models import build_detector, forbid_external_weights
from metrics import detection_ap50
from audit_positive_sampling import proposals, max_iou
from train import fingerprint


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('reports/training/object-reserve-01/proposal-comparison'))
    out=parser.parse_args().output
    out.mkdir(exist_ok=False)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    experiment=read(Path('reports/training/object-reserve-01/comparison.json'))
    torch.set_num_threads(4);forbid_external_weights();device='cuda' if torch.cuda.is_available() else 'cpu'
    results=[]
    for pair in experiment['records']:
        for variant,run in [('control',pair['control_run']),('reserve',pair['run'])]:
            path=Path('runs')/run/'last.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest()
            ck=torch.load(path,map_location='cpu',weights_only=True);c=ck['config']
            assert ck['optimizer_updates']==1950 and c['initialization']=='random' and not c['pretrained']
            root=Path(c['dataset'])
            for name,h in read(root/'dataset.lock.json').items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
            ds=SeparateHelmetDataset(root,'val',augment=False)
            model=build_detector(5,c['feature_stride'],c['proposal_multiplier']).to(device)
            model.load_state_dict(ck['model']);model.eval();before=fingerprint(model)
            heads=[];predictions=[];targets=[]
            for start in range(0,len(ds),2):
                samples=[ds[i] for i in range(start,min(start+2,len(ds)))]
                xs=[x.to(device) for x,_ in samples];ts=[{k:v.to(device) for k,v in t.items()} for _,t in samples]
                pred=model(xs);predictions.extend([{k:v.cpu() for k,v in p.items()} for p in pred]);targets.extend([t for _,t in samples])
                ims,truth=model.transform(xs,ts);features=OrderedDict([('0',model.backbone(ims.tensors))])
                _,boxes,stages,obj=proposals(model,ims,features)
                pooled=model.roi_heads.box_roi_pool(features,boxes,ims.image_sizes)
                logits,deltas=model.roi_heads.box_predictor(model.roi_heads.box_head(pooled))
                probs=logits.softmax(-1).split([len(b) for b in boxes])
                refined=model.roi_heads.box_coder.decode(deltas,boxes).split([len(b) for b in boxes])
                gt_pool=model.roi_heads.box_roi_pool(features,[t['boxes'] for t in truth],ims.image_sizes)
                gt_logits,_=model.roi_heads.box_predictor(model.roi_heads.box_head(gt_pool))
                gt_probs=gt_logits.softmax(-1).split([len(t['boxes']) for t in truth])
                for i,t in enumerate(truth):
                    row=ds.rows[start+i]
                    for j,label in enumerate(t['labels'].tolist()):
                        if label not in (3,4):continue
                        gt=t['boxes'][j:j+1];ov=box_iou(gt,boxes[i])[0];covered=ov>=.5
                        decoded_ov=box_iou(gt,stages[i]['decoded_clipped'])[0];cover_all=decoded_ov>=.5
                        score=obj[i][cover_all].max() if cover_all.any() else None
                        original=samples[i][1]['boxes'][j:j+1].to(device);correct=pred[i]['labels']==label
                        best=int(ov.argmax()) if len(ov) else None
                        refined_boxes=clip_boxes_to_image(refined[i][:,label,:],ims.image_sizes[i])
                        postbox=box_iou(gt,refined_boxes)[0]
                        covered_indices=torch.where(covered)[0]
                        top_correct=int(covered_indices[probs[i][covered,label].argmax()]) if covered.any() else None
                        heads.append({'image_id':row['image_id'],'entity_id':row['annotations'][j]['entity_id'],'label':label,
                            'model_image_size':list(ims.image_sizes[i]),'gt_model_box':gt[0].tolist(),
                            'highest_scoring_covering_proposal':None if top_correct is None else {
                                'correct_probability':float(probs[i][top_correct,label]),
                                'box_before_regression':boxes[i][top_correct].tolist(),
                                'box_after_regression':refined_boxes[top_correct].tolist(),
                                'iou_before':float(ov[top_correct]),'iou_after':float(postbox[top_correct])},
                            'stage_iou':{k:float(max_iou(gt,b)[0]) for k,b in stages[i].items()},
                            'best_covering_rank':int((obj[i]>score).sum())+1 if score is not None else None,
                            'covering_proposals':int(covered.sum()),
                            'covering_argmax_classes':torch.bincount(probs[i][covered].argmax(1),minlength=5).tolist(),
                            'covering_max_correct_probability':float(probs[i][covered,label].max()) if covered.any() else None,
                            'best_overlap_probabilities':probs[i][best].tolist() if best is not None else None,
                            'gt_box_probabilities':gt_probs[i][j].tolist(),
                            'correct_regressed_iou_before_threshold':float(postbox.max()) if len(postbox) else 0,
                            'correct_candidates_iou50_score05':int(((postbox>=.5)&(probs[i][:,label]>.05)).sum()),
                            'correct_candidates_iou50_score50':int(((postbox>=.5)&(probs[i][:,label]>=.5)).sum()),
                            'final_correct_iou':float(max_iou(original,pred[i]['boxes'][correct])[0]),
                            'final_correct_iou_score50':float(max_iou(original,pred[i]['boxes'][correct&(pred[i]['scores']>=.5)])[0])})
            metric=detection_ap50(predictions,targets,(1,2,3,4))
            assert abs(metric['map50']-ck['validation']['map50'])<1e-6
            assert fingerprint(model)==before and hashlib.sha256(path.read_bytes()).hexdigest()==digest
            result={'run':run,'variant':variant,'images':pair['images'],'seed':pair['seed'],
                    'checkpoint_sha256':digest,'model_state_unchanged':True,'metrics_reproduced':metric,'heads':heads}
            (out/(run+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');results.append(result)
            print(run+' verified',flush=True)
            del ck,model
    (out/'comparison.json').write_text(json.dumps({'results':results,
        'method':'Eval mode, same manifest order and batch 2 as checkpoint evaluation. All 9 validation heads. No training, test, threshold changes or GT injection into normal predictions. GT-box classification is a separate diagnostic.',
        'class_probabilities_order':['background','motorcycle','occupant','helmet','nohelmet'],
        'coverage_warning':'Per-reference coverage is not one-to-one recall. Regressed candidates are before final NMS and maximum detection limit.'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
