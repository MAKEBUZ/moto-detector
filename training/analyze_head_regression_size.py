"""Paired per-head IoU changes by class and internal pixel size; frozen weights."""
import hashlib
import json
import statistics
from collections import OrderedDict, defaultdict
from pathlib import Path
import torch
from torchvision.ops import box_iou, clip_boxes_to_image
from data import SeparateHelmetDataset
from models import build_detector, forbid_external_weights
from train import fingerprint


def summarize(rows, policy):
    selected=[r[policy] for r in rows if r[policy] is not None]
    deltas=[r['delta_iou'] for r in selected]
    return {'references':len(rows),'unique_heads':len({(r['image_id'],r['entity_id']) for r in rows}),
            'eligible':len(selected),'missing_proposal':len(rows)-len(selected),
            'mean_before':statistics.mean(r['iou_before'] for r in selected) if selected else None,
            'mean_after':statistics.mean(r['iou_after'] for r in selected) if selected else None,
            'mean_delta':statistics.mean(deltas) if deltas else None,
            'median_delta':statistics.median(deltas) if deltas else None,
            'improved_gt01':sum(d>.01 for d in deltas),'worsened_gt01':sum(d<-.01 for d in deltas),
            'within01':sum(abs(d)<=.01 for d in deltas),
            'gain_iou50':sum(r['iou_before']<.5<=r['iou_after'] for r in selected),
            'loss_iou50':sum(r['iou_after']<.5<=r['iou_before'] for r in selected),
            'score50_eligible':sum(r['score']>=.5 for r in selected),
            'score50_gain_iou50':sum(r['score']>=.5 and r['iou_before']<.5<=r['iou_after'] for r in selected),
            'score50_loss_iou50':sum(r['score']>=.5 and r['iou_after']<.5<=r['iou_before'] for r in selected)}


@torch.inference_mode()
def main():
    out=Path('reports/training/object-reserve-01/regression-by-size')
    out.mkdir(exist_ok=False)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    pairs=read(Path('reports/training/object-reserve-01/comparison.json'))['records']
    torch.set_num_threads(4);forbid_external_weights();device='cuda' if torch.cuda.is_available() else 'cpu'
    records=[];checks=[]
    for pair in pairs:
        for variant,run in [('control',pair['control_run']),('reserve',pair['run'])]:
            checkpoint=Path('runs')/run/'last.pt';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            ck=torch.load(checkpoint,map_location='cpu',weights_only=True);c=ck['config'];root=Path(c['dataset'])
            assert ck['optimizer_updates']==1950 and not c['pretrained'] and c['initialization']=='random'
            for name,h in read(root/'dataset.lock.json').items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
            model=build_detector(5,c['feature_stride'],c['proposal_multiplier']).to(device)
            model.load_state_dict(ck['model']);model.eval();initial=fingerprint(model)
            local=[]
            for split in ('train','val'):
                ds=SeparateHelmetDataset(root,split,augment=False)
                for start in range(0,len(ds),2):
                    samples=[ds[i] for i in range(start,min(start+2,len(ds)))]
                    xs=[x.to(device) for x,_ in samples];ts=[{k:v.to(device) for k,v in t.items()} for _,t in samples]
                    ims,truth=model.transform(xs,ts)
                    features=OrderedDict([('0',model.backbone(ims.tensors))]);proposals,_=model.rpn(ims,features)
                    roi=model.roi_heads;pooled=roi.box_roi_pool(features,proposals,ims.image_sizes)
                    logits,deltas=roi.box_predictor(roi.box_head(pooled))
                    probs=logits.softmax(-1).split([len(p) for p in proposals]);decoded=roi.box_coder.decode(deltas,proposals).split([len(p) for p in proposals])
                    for i,t in enumerate(truth):
                        meta=ds.rows[start+i];pre=box_iou(t['boxes'],proposals[i])
                        for j,cls in enumerate(t['labels'].tolist()):
                            if cls not in (3,4):continue
                            box=t['boxes'][j];side=float((box[2:]-box[:2]).min());bucket='small_lt16' if side<16 else 'medium_16to32' if side<32 else 'large_ge32'
                            adjusted=clip_boxes_to_image(decoded[i][:,cls,:],ims.image_sizes[i])
                            post=box_iou(box[None],adjusted)[0]
                            def select(min_iou):
                                candidates=torch.where(pre[j]>=min_iou)[0]
                                if not len(candidates):return None
                                k=int(candidates[probs[i][candidates,cls].argmax()])
                                score=float(probs[i][k,cls]);before=float(pre[j,k]);after=float(post[k])
                                return {'proposal_index':k,'score':score,'iou_before':before,'iou_after':after,'delta_iou':after-before,
                                        'box_before':proposals[i][k].tolist(),'box_after':adjusted[k].tolist()}
                            local.append({'run':run,'variant':variant,'train_images':pair['images'],'seed':pair['seed'],'split':split,
                                'image_id':meta['image_id'],'entity_id':meta['annotations'][j]['entity_id'],
                                'class_name':'helmet' if cls==3 else 'nohelmet','size_bucket':bucket,'min_side_model_px':side,
                                'gt_box':box.tolist(),'model_image_size':list(ims.image_sizes[i]),
                                'near03':select(.3),'covered05':select(.5)})
            assert fingerprint(model)==initial and hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
            checks.append({'run':run,'checkpoint_sha256':digest,'state_unchanged':True,'references':len(local)})
            records.extend(local);(out/(run+'.json')).write_text(json.dumps(local,indent=2),encoding='utf-8')
            print(run+' done',flush=True)
            del model,ck
    pooled=[];per_run=[]
    for policy in ('near03','covered05'):
        groups=defaultdict(list);runs=defaultdict(list)
        for r in records:
            groups[(r['split'],r['class_name'],r['size_bucket'])].append(r)
            runs[(r['run'],r['split'],r['class_name'],r['size_bucket'])].append(r)
        for key,rows in sorted(groups.items()):pooled.append(dict(zip(('split','class_name','size_bucket'),key))|{'policy':policy}|summarize(rows,policy))
        for key,rows in sorted(runs.items()):per_run.append(dict(zip(('run','split','class_name','size_bucket'),key))|{'policy':policy}|summarize(rows,policy))
    result={'pooled':pooled,'per_run':per_run,'checks':checks,
        'selection':'Per GT head, select highest correct-class score among FINAL RPN proposals with PRE-regression IoU >=0.3 (primary) or >=0.5 (sensitivity). Keep the same proposal index, class and score when evaluating adjusted box. No selection by post-regression IoU.',
        'sizes':'Minimum GT box side in actual model-transformed pixels, <16, [16,32), >=32. Not COCO area categories.',
        'method':'Eight final checkpoints, eval mode for both train/val, batch2 manifest order, no augmentation, no GT injection. Before final score threshold/NMS. Train is in-sample diagnosis; val is small and repeatedly consulted.',
        'limitations':'GT-guided conditional localization analysis, not AP/precision/recall. Missing proposals reported separately. Pooled counts repeat the same GT across checkpoints; not independent samples or statistical evidence. Head can share a proposal with another head.',
        'training_performed':False,'test_used':False}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
