"""Frozen old/new validation comparison; no training or threshold selection."""
import hashlib
import argparse
import json
from pathlib import Path
import torch
from torchvision.ops import box_iou
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageOps
from data import SeparateHelmetDataset
from models import build_detector, forbid_external_weights
from metrics import detection_ap50


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/prepared/helmet-separate-v2-expanded-02'))
    parser.add_argument('--old-run', default='detector-separate-v2-proposals4-01')
    parser.add_argument('--new-run', default='detector-separate-v2-expanded-02')
    parser.add_argument('--output', type=Path, default=Path('reports/training/separate-v2-expanded-02/nohelmet-review'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    forbid_external_weights()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = args.dataset
    for name, digest in json.loads((root/'dataset.lock.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    ds = SeparateHelmetDataset(root, 'val', augment=False)
    samples = [ds[i] for i in range(len(ds))]
    targets = [t for _, t in samples]
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    reports = {}; all_preds = {}
    for tag, run in [('old', args.old_run), ('new', args.new_run)]:
        checkpoint = Path('runs')/run/'best.pt'
        ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
        c = ck['config']
        assert c['initialization'] == 'random' and c['pretrained'] is False
        assert (Path(c['dataset'])/'val-manifest.jsonl').read_bytes() == (root/'val-manifest.jsonl').read_bytes()
        model = build_detector(5, c['feature_stride'], c['proposal_multiplier'])
        model.load_state_dict(ck['model']); model.to(device).eval()
        preds = []
        # Preserve evaluation batch size and padding, matching checkpoint selection.
        for start in range(0, len(samples), 2):
            preds.extend([{k:v.cpu() for k,v in p.items()} for p in model([x.to(device) for x,_ in samples[start:start+2]])])
        metrics = detection_ap50(preds, targets, (1,2,3,4))
        assert abs(metrics['map50'] - ck['validation']['map50']) < 1e-6
        entries = sorted([(float(p['scores'][j]),i,j) for i,p in enumerate(preds) for j in range(len(p['scores'])) if int(p['labels'][j])==4], reverse=True)
        used = [set() for _ in samples]; ranked = []; tp = 0
        for rank, (score,i,j) in enumerate(entries,1):
            p = preds[i]; t = targets[i]; overlaps = box_iou(p['boxes'][j:j+1],t['boxes'])[0]
            candidates = sorted([k for k in range(len(t['labels'])) if int(t['labels'][k])==4],key=lambda k:-float(overlaps[k]))
            match = next((k for k in candidates if k not in used[i] and overlaps[k]>=.5),None)
            if match is not None: used[i].add(match);tp+=1
            any_k = int(overlaps.argmax())
            reason = 'TP' if match is not None else ('duplicate' if any(float(overlaps[k])>=.5 for k in candidates) else 'wrong_class' if float(overlaps[any_k])>=.5 else 'localization_or_background')
            ranked.append({'rank':rank,'image_id':ds.rows[i]['image_id'],'prediction_index':j,'score':score,'box':p['boxes'][j].tolist(),'matched_entity':ds.rows[i]['annotations'][match]['entity_id'] if match is not None else None,'reason':reason,'max_gt_iou':float(overlaps[any_k]),'precision':tp/rank,'recall':tp/6})
        heads = []
        for i,t in enumerate(targets):
            p = preds[i]
            for j,label in enumerate(t['labels']):
                if int(label)!=4: continue
                overlaps = box_iou(t['boxes'][j:j+1],p['boxes'])[0]
                def best(cls):
                    inds = [k for k in range(len(p['labels'])) if int(p['labels'][k])==cls]
                    if not inds:return None
                    k=max(inds,key=lambda k:float(overlaps[k]))
                    return {'iou':float(overlaps[k]),'score':float(p['scores'][k]),'box':p['boxes'][k].tolist()}
                heads.append({'image_id':ds.rows[i]['image_id'],'entity_id':ds.rows[i]['annotations'][j]['entity_id'],'box':t['boxes'][j].tolist(),'nohelmet':best(4),'helmet':best(3)})
        reports[tag]={'run':run,'epoch':ck['epoch'],'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'metrics':metrics,'ranked_nohelmet':ranked,'heads':heads,
                      'counts':{str(th):{'tp':sum(r['score']>=th and r['reason']=='TP' for r in ranked),'fp':sum(r['score']>=th and r['reason']!='TP' for r in ranked),'fn':6-sum(r['score']>=th and r['reason']=='TP' for r in ranked)} for th in (.05,.5)},
                      'predictions':[{k:v.tolist() for k,v in p.items()} for p in preds]}
        all_preds[tag]=preds
        del model
    # Context crops: ground truth green, no-helmet prediction red, helmet prediction blue.
    sheet=Image.new('RGB',(900,6*250),'white');draw=ImageDraw.Draw(sheet)
    for n,head in enumerate(reports['new']['heads']):
        i=next(i for i,r in enumerate(ds.rows) if r['image_id']==head['image_id'])
        base=to_pil_image(samples[i][0]);box=head['box'];x0,y0,x1,y1=box
        margin=max(x1-x0,y1-y0)*.6
        crop=(max(0,int(x0-margin)),max(0,int(y0-margin)),min(base.width,int(x1+margin)),min(base.height,int(y1+margin)))
        for col,tag in enumerate(('truth','old','new')):
            im=base.copy();d=ImageDraw.Draw(im);d.rectangle(box,outline='lime',width=2)
            title=head['image_id']+' '+head['entity_id']+' '+tag
            if tag!='truth':
                h=reports[tag]['heads'][n]
                for kind,color in [('nohelmet','red'),('helmet','blue')]:
                    q=h[kind]
                    if q and q['iou']>.1:d.rectangle(q['box'],outline=color,width=2)
                q=h['nohelmet']
                if q:title+=f" IoU {q['iou']:.2f} s {q['score']:.2f}"
            tile=ImageOps.contain(im.crop(crop),(290,215))
            sheet.paste(tile,(col*300,n*250+30));draw.text((col*300+3,n*250+4),title,fill='black')
    sheet.save(out/'comparison.jpg')
    (out/'comparison.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps({tag:{'counts':r['counts'],'heads':r['heads'],'ranked':r['ranked_nohelmet'][:10]} for tag,r in reports.items()},indent=2))


if __name__=='__main__':main()
