"""Rank train-only review candidates; never generate confirmed negative labels."""
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms.functional import pil_to_tensor
from torchvision.ops import box_iou
from models import build_detector, forbid_external_weights
from train import fingerprint


@torch.inference_mode()
def main():
    out=Path('reports/separate-v2/hard-negatives-selection')
    destination=out/'mining.json'
    if destination.exists():raise FileExistsError(destination)
    rows=json.loads((out/'candidates.json').read_text())
    decisions={r['image']:r for r in map(json.loads,Path('reports/preparation/image-decisions.jsonl').read_text().splitlines())}
    previous=[json.loads(p.read_text()) for p in Path('data/annotations').glob('*/draft.json')]
    excluded={s['group_id'] for b in previous for s in b['scenes']}
    excluded_hashes={s['sha256'] for b in previous for s in b['scenes']}
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('data/prepared').glob('*/val*') if p.is_file()}
    for row in rows:
        d=decisions[row['image']]
        assert d['split']=='train' and d['groupId'] not in excluded and row['sha256'] not in excluded_hashes
        assert hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()==row['sha256']
    jobs=[('detector-reserve1950-n44-s20261125','last.pt'),('detector-separate-v2-expanded-02','best.pt')]
    protocol={'jobs':jobs,'rpn_pre_nms':4000,'minimum_head_score':.05,'maximum_source_head_iou':.1,
              'source_labels_are_unverified':True,'training':False,'selection_split':'train','excluded_prior_groups':sorted(excluded),
              'protected_validation_hashes':protected,'purpose':'Review queue only: low overlap is not proof of background.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.set_num_threads(4);forbid_external_weights();device='cuda' if torch.cuda.is_available() else 'cpu'
    records=[]
    for run,filename in jobs:
        path=Path('runs')/run/filename;digest=hashlib.sha256(path.read_bytes()).hexdigest()
        ck=torch.load(path,map_location='cpu',weights_only=True);c=ck['config'];assert not c['pretrained'] and c['initialization']=='random'
        model=build_detector(5,c['feature_stride'],c['proposal_multiplier']).to(device);model.load_state_dict(ck['model']);model.eval();before=fingerprint(model)
        model.rpn._pre_nms_top_n['testing']=4000
        for start in range(0,len(rows),2):
            batch=rows[start:start+2];xs=[];scales=[]
            for row in batch:
                im=Image.open(row['image']).convert('RGB');w,h=im.size;s=min(1,640/max(w,h));nw,nh=max(1,round(w*s)),max(1,round(h*s))
                xs.append(pil_to_tensor(im.resize((nw,nh),Image.Resampling.BILINEAR)).float().div(255).to(device));scales.append((nw/w,nh/h))
            predictions=model(xs)
            for row,p,(sx,sy) in zip(batch,predictions,scales):
                source=torch.tensor([a['bbox'] for a in row['annotations'] if a['class'] in ('Helmet','NoHelmet')],device=device,dtype=torch.float32)
                boxes=p['boxes']/torch.tensor([sx,sy,sx,sy],device=device)
                for j in torch.where((p['labels']>=3)&(p['scores']>=.05))[0].tolist():
                    overlap=float(box_iou(boxes[j:j+1],source).max())
                    if overlap>=.1:continue
                    records.append({'image_id':Path(row['image']).stem,'image':row['image'],'group_id':decisions[row['image']]['groupId'],
                                    'run':run,'checkpoint_sha256':digest,'box':boxes[j].tolist(),'score':float(p['scores'][j]),'label':int(p['labels'][j]),
                                    'max_source_head_iou':overlap,'review_status':'pending'})
        assert fingerprint(model)==before and hashlib.sha256(path.read_bytes()).hexdigest()==digest
        del model,ck
    for name,digest in protected.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    records.sort(key=lambda r:-r['score']);destination.write_text(json.dumps(records,indent=2),encoding='utf-8')
    chosen=[];seen=set()
    for group in sorted({decisions[r['image']]['groupId'] for r in rows}):
        count=0
        for r in records:
            if r['group_id']==group and r['image_id'] not in seen:
                chosen.append(r);seen.add(r['image_id']);count+=1
                if count==3:break
    (out/'visual-candidates.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8')
    for page in range((len(chosen)+8)//9):
        sheet=Image.new('RGB',(1200,1080),'white');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(chosen[page*9:page*9+9]):
            im=Image.open(r['image']).convert('RGB');ImageDraw.Draw(im).rectangle(r['box'],outline='red',width=4)
            thumb=ImageOps.contain(im,(390,310));x=(j%3)*400;y=(j//3)*360;sheet.paste(thumb,(x,y+45))
            draw.text((x+4,y+3),f"{r['image_id']} score={r['score']:.3f} class={r['label']}\n{r['group_id']}",fill='black')
        sheet.save(out/f'mined-{page+1}.jpg')
    print(json.dumps({'candidate_images':len(rows),'groups':len({decisions[r['image']]['groupId'] for r in rows}),'regions':len(records),'visual_candidates':len(chosen),'validation_unchanged':True}))


if __name__=='__main__':main()
