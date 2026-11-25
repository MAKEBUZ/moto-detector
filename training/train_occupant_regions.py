"""Tune the occupant logit on inspected parts while replaying reviewed ground truth."""
import hashlib
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor

from data import SeparateHelmetDataset, collate_detection
from predict import load_model
from train_joint_clean import evaluate


@torch.no_grad()
def features(model, im, boxes):
    w,h=im.size
    scale=min(1,640/max(w,h))
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    tensor=pil_to_tensor(im.resize((nw,nh),Image.Resampling.BILINEAR)).cuda().float()/255
    batch,_=model.transform([tensor])
    H,W=batch.image_sizes[0]
    boxes=torch.tensor(boxes,device="cuda")*tensor.new_tensor([W/w,H/h,W/w,H/h])
    maps=OrderedDict([("0",model.backbone(batch.tensors))])
    pool=model.roi_heads.box_roi_pool(maps,[boxes],batch.image_sizes)
    return model.roi_heads.box_head(pool)


def main():
    parent="joint-reviewed-08-10"
    folder=Path("runs/joint-occupant-negatives-11")
    folder.mkdir(exist_ok=False)
    model,cfg,*_=load_model(parent)
    region_path=Path("reports/training/occupant-negatives-01/selected.json")
    selected=json.loads(region_path.read_text())
    regions=defaultdict(list)
    for r in selected:
        regions[(r["ds"],r["image"],tuple(r.get("crop",[])))].append(r["box"])
    banks=defaultdict(list)
    base="helmet-separate-v2-expanded-09"
    def add(im, boxes, kinds):
        for flip in (False,True):
            bb=[[im.width-r,t,im.width-l,b] for l,t,r,b in boxes] if flip else boxes
            feats=features(model,ImageOps.mirror(im) if flip else im,bb)
            for k,f in zip(kinds,feats):banks[k].append(f)
    for ds in (base,"reviewed-08-addon"):
        root=Path("data/prepared")/ds
        for line in (root/"train-manifest.jsonl").read_text().splitlines():
            r=json.loads(line)
            assert r["split"]=="train"
            with Image.open(root/r["image"]) as source:im=source.convert("RGB")
            boxes=[a["box"] for a in r["annotations"]]
            kinds=["person" if a["classId"]==1 else "other" for a in r["annotations"]]
            add(im,boxes,kinds)
            for key in [k for k in regions if k[:2]==(ds,r["image"])]:
                rect=key[2];bs=regions.pop(key)
                add(im.crop(rect) if rect else im,bs,["part"]*len(bs))
    assert not regions
    head=model.roi_heads.box_predictor.cls_score
    index=2  # background=0, motorcycle=1, occupant=2
    initial=head.weight[index].detach().clone()
    bias=head.bias[index].detach().clone()
    banks={k:torch.stack(v) for k,v in banks.items()}
    with torch.no_grad():
        logits={k:head(v) for k,v in banks.items()}
        margins={k:l[:,index]-torch.logsumexp(l[:,[0,1,3,4]],1) for k,l in logits.items()}
    delta=torch.nn.Parameter(torch.zeros_like(initial));offset=torch.nn.Parameter(torch.zeros_like(bias))
    optimizer=torch.optim.Adam([delta,offset],lr=.00035)
    validation=DataLoader(SeparateHelmetDataset(Path("data/prepared")/base,"val",augment=False),batch_size=2,collate_fn=collate_detection)
    parent_path=Path("runs")/parent/"best.pt"
    cfg=dict(cfg,initialization="occupant_region_transfer",parent_run=parent,run=folder.name,
             parent_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest())
    history=[];best=None
    for step in range(1,101):
        shift={k:x@delta+offset for k,x in banks.items()}
        loss=(F.softplus(margins["part"]+shift["part"]).mean()
              + F.softplus(-margins["person"]-shift["person"]).mean()
              + .1*shift["other"].square().mean() + .05*shift["person"].square().mean())
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%10:continue
        with torch.no_grad():
            head.weight[index].copy_(initial+delta);head.bias[index].copy_(bias+offset)
        result=evaluate(model,validation);c=result["counts"][1]
        rank=(2*c["tp"]/(2*c["tp"]+c["fp"]+c["fn"]),result["metrics"]["map50"])
        history.append(dict(step=step,loss=float(loss.detach()),**result))
        if best is None or rank>best:
            best=rank
            torch.save(dict(model=model.state_dict(),config=cfg,epoch=step,validation=result["metrics"]),folder/"best.pt")
    (folder/"history.json").write_text(json.dumps(history,indent=2))
    (folder/"protocol.json").write_text(json.dumps(dict(parent=parent,regions=len(selected),images=83,
        classes={k:len(v) for k,v in banks.items()},selection="validation occupant F1 then mAP50; earliest tie",
        regions_sha256=hashlib.sha256(region_path.read_bytes()).hexdigest()),indent=2))
    print(best,flush=True)

if __name__=="__main__":main()
