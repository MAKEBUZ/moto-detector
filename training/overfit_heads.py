"""Train-only memorization diagnostic, always initialized randomly."""
import hashlib
import json
import random
import time
from collections import OrderedDict, Counter
from pathlib import Path
import torch
from torchvision.ops import box_iou
from data import SeparateHelmetDataset
from models import build_detector, forbid_external_weights
from metrics import detection_ap50
from train import fingerprint


@torch.inference_mode()
def probe(model, samples, device):
    model.eval(); rows = []; predictions = []; targets = []
    for image, target in samples:
        image = image.to(device)
        pred = model([image])[0]
        predictions.append({k:v.cpu() for k,v in pred.items()}); targets.append(target)
        ims, ts = model.transform([image], [{k:v.to(device) for k,v in target.items()}])
        features = OrderedDict([('0', model.backbone(ims.tensors))])
        proposals, _ = model.rpn(ims, features, ts)
        anchors = model.rpn.anchor_generator(ims, list(features.values()))
        anchor_labels, _ = model.rpn.assign_targets_to_anchors(anchors, ts)
        gt = ts[0]
        pooled = model.roi_heads.box_roi_pool(features, [gt['boxes']], ims.image_sizes)
        logits, _ = model.roi_heads.box_predictor(model.roi_heads.box_head(pooled))
        for j, label in enumerate(target['labels'].tolist()):
            if label not in (3,4): continue
            box = target['boxes'][j:j+1].to(device)
            mask = pred['labels'] == label
            active = mask & (pred['scores'] >= .5)
            anchor_iou = box_iou(gt['boxes'][j:j+1], anchors[0])[0]
            rows.append({'label':label, 'image_id':int(target['image_id']),
                         'anchor_max_iou':float(anchor_iou.max()),
                         'positive_anchors_overlapping_head_iou50':int(((anchor_iou>=.5)&(anchor_labels[0]==1)).sum()),
                         'rpn_max_iou':float(box_iou(gt['boxes'][j:j+1], proposals[0]).max()) if len(proposals[0]) else 0.,
                         'correct_class_max_iou':float(box_iou(box,pred['boxes'][mask]).max()) if mask.any() else 0.,
                         'correct_class_max_iou_score50':float(box_iou(box,pred['boxes'][active]).max()) if active.any() else 0.,
                         'oracle_predicted_class':int(logits[j].argmax())})
    return {'head_count':len(rows), 'rpn_covered_iou50':sum(r['rpn_max_iou']>=.5 for r in rows),
            'correct_head_coverage_iou50':sum(r['correct_class_max_iou']>=.5 for r in rows),
            'correct_head_coverage_iou50_score50':sum(r['correct_class_max_iou_score50']>=.5 for r in rows),
            'oracle_correct':sum(r['oracle_predicted_class']==r['label'] for r in rows),
            'head_ap50':detection_ap50(predictions,targets,(3,4)), 'heads':rows}


def main():
    root = Path('data/prepared/helmet-separate-v2-expanded-01')
    out = Path('runs/detector-head-overfit-01')
    if out.exists(): raise FileExistsError(out)
    for name,digest in json.loads((root/'dataset.lock.json').read_text()).items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest: raise ValueError('Dataset changed')
    ds = SeparateHelmetDataset(root,'train',augment=False)
    # Two relatively large examples per head class, picked without model scores.
    chosen = []
    for cls in (2,3):
        def rank(i):
            r=ds.rows[i]; scale=min(1.,640/max(r['width'],r['height']))
            return max((min(a['box'][2]-a['box'][0],a['box'][3]-a['box'][1])*scale for a in r['annotations'] if a['classId']==cls),default=0)
        eligible=[i for i in range(len(ds)) if rank(i)>0 and i not in chosen]
        chosen.extend(sorted(eligible,key=lambda i:(-rank(i),ds.rows[i]['image_id']))[:2])
    if len(chosen)!=4: raise ValueError('Need four distinct images')
    samples=[ds[i] for i in chosen]
    random.seed(20261124); torch.manual_seed(20261124); torch.set_num_threads(4)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(20261124)
    forbid_external_weights(); device='cuda' if torch.cuda.is_available() else 'cpu'
    model=build_detector(5); initial=fingerprint(model); model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    out.mkdir(parents=True)
    config={'initialization':'random','pretrained':False,'initial_state_sha256':initial,'seed':20261124,
            'steps':300,'batch_size':2,'learning_rate':.0003,'scheduler':None,'augmentation':False,'device':device,
            'sample_images':[ds.rows[i]['image_id'] for i in chosen], 'selection':'two largest source-scaled heads per class, distinct train images',
            'all_four_classes_retained':True,'validation_used':False,'test_used':False,
            'dataset_lock_sha256':hashlib.sha256((root/'dataset.lock.json').read_bytes()).hexdigest(),
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path('training/models.py'),Path('training/data.py'),Path('training/metrics.py'))}}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    history=[{'step':0,'diagnostic':probe(model,samples,device)}]; losses=[]; start=time.time()
    print(json.dumps({'step':0,'sample_images':config['sample_images'],'diagnostic':{k:v for k,v in history[0]['diagnostic'].items() if k!='heads'}}),flush=True)
    for step in range(1,301):
        model.train(); indices=[(2*(step-1)+j)%4 for j in range(2)]
        images=[samples[i][0].to(device) for i in indices]
        targets=[{k:v.to(device) for k,v in samples[i][1].items()} for i in indices]
        optimizer.zero_grad(set_to_none=True); parts=model(images,targets); loss=sum(parts.values())
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
        losses.append({'step':step,**{k:float(v.detach()) for k,v in parts.items()}})
        if step%25==0:
            diagnostic=probe(model,samples,device)
            averages={k:sum(r[k] for r in losses[-25:])/25 for k in parts}
            record={'step':step,'mean_losses_last25':averages,'diagnostic':diagnostic,'seconds':time.time()-start}
            history.append(record); (out/'history.json').write_text(json.dumps(history,indent=2))
            print(json.dumps({**record,'diagnostic':{k:v for k,v in diagnostic.items() if k!='heads'}}),flush=True)
    torch.save({'model':model.state_dict(),'config':config,'step':300},out/'last.pt')
    (out/'losses.json').write_text(json.dumps(losses,indent=2))
    result={'initial_state_sha256':initial,'final_state_sha256':fingerprint(model),'steps':300,'seconds':time.time()-start,
            'training_sample_only':True,'validation_used':False,'test_used':False,'final_diagnostic':history[-1]['diagnostic']}
    assert result['initial_state_sha256']!=result['final_state_sha256']
    (out/'result.json').write_text(json.dumps(result,indent=2))
    report=Path('reports/training/head-overfit-01'); report.mkdir(parents=True,exist_ok=True)
    for name in ('config.json','history.json','losses.json','result.json'):
        (report/name).write_bytes((out/name).read_bytes())
    print('Completed train-only overfit diagnostic',flush=True)


if __name__=='__main__': main()
