"""Ajusta las dos clases de cabeza con los frontales revisados del conjunto de entrenamiento."""
from pathlib import Path
import json, hashlib
import torch
from PIL import Image,ImageOps
from torch.utils.data import DataLoader
from torch.nn import functional as F
from train_occupant_regions import features
from train_joint_clean import evaluate
from data import SeparateHelmetDataset,collate_detection
from predict import load_model


def main():
    src=Path('runs/frontal-scratch-01');dest=Path('runs/frontal-scratch-frontales-02')
    if not (src/'result.json').exists():
        raise RuntimeError('Complete el entrenamiento frontal-scratch-01 antes del ajuste.')
    dest.mkdir(exist_ok=False)
    model,cfg,*_=load_model(src.name)
    assert cfg['initialization']=='random' and cfg['pretrained']==False
    root=Path(cfg['dataset'])
    rows=[json.loads(l) for l in (root/'train-manifest.jsonl').read_text().splitlines()]
    review=json.loads(Path('data/annotations/visores-frontales-01/visores-frontales-01-reviewed.json').read_text(encoding='utf-8'))
    regions={s['image_id']:s['frontal_negativo'] for s in review['scenes']}
    bank=[]; labels=[]; neg=[]
    for r in rows:
     with Image.open(root/r['image']) as im: image=im.convert('RGB')
     boxes=[a['box'] for a in r['annotations']];cats=[a['classId']+1 for a in r['annotations']]
     extra=regions.get(r['image_id'])
     if extra: boxes=boxes+[extra]
     for flip in [False,True]:
      b=[[image.width-R,t,image.width-l,B] for l,t,R,B in boxes] if flip else boxes
      x=features(model,ImageOps.mirror(image) if flip else image,b)
      bank.append(x[:len(cats)]);labels.extend(cats)
      if extra: neg.append(x[-1])
    assert len(neg)==16
    x=torch.cat(bank);y=torch.tensor(labels,device=x.device);n=torch.stack(neg)
    head=model.roi_heads.box_predictor.cls_score
    weights=head.weight.detach().clone();bias=head.bias.detach().clone();base=head(x).detach()
    dw=torch.nn.Parameter(torch.zeros_like(weights[3:5]));db=torch.nn.Parameter(torch.zeros_like(bias[3:5]))
    opt=torch.optim.Adam([dw,db],lr=.0001)
    def logits(z):
     old=F.linear(z,weights,bias);return torch.cat([old[:,:3],old[:,3:]+F.linear(z,dw,db)],1)
    val=DataLoader(SeparateHelmetDataset(root,'val',augment=False),batch_size=2,collate_fn=collate_detection)
    history=[];best=-1
    for step in range(1,61):
     l=logits(x);front=logits(n);loss=F.cross_entropy(l,y)+F.softplus(torch.logsumexp(front[:,3:],1)-torch.logsumexp(front[:,:3],1)).mean()+.15*(l-base).square().mean()
     opt.zero_grad();loss.backward();opt.step()
     if step%10:continue
     with torch.no_grad():head.weight[3:5].copy_(weights[3:5]+dw);head.bias[3:5].copy_(bias[3:5]+db)
     result=evaluate(model,val);score=result['metrics']['map50'];history.append(dict(step=step,loss=loss.item(),map50=score))
     if score>best:
      best=score;torch.save(dict(model=model.state_dict(),config=dict(cfg,run=dest.name,calibration_parent=src.name,calibration_parent_sha256=hashlib.sha256((src/'best.pt').read_bytes()).hexdigest()),epoch=step),dest/'best.pt')
    out=Path('reports/training')/dest.name;out.mkdir(exist_ok=True)
    (out/'calibration.json').write_text(json.dumps(dict(regions=regions,history=history),indent=2))
    print(history,flush=True)


if __name__ == "__main__":
    main()
