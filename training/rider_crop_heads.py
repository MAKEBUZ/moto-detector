"""Head detection inside rider crops; train and infer in the same coordinate system."""
import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import DataLoader, WeightedRandomSampler
from data import collate_detection
from predict import load_model, detector_objects
from train_joint_clean import evaluate
from photo_results import box_overlap, summarize


def region(box):
    l,t,r,b=box
    side=max(32, min(.85*(b-t),1.4*(r-l)))
    cx,cy=(l+r)/2,t+.16*(b-t)
    return (round(cx-side/2),round(cy-side/2),round(cx+side/2),round(cy+side/2))


class Crops:
    def __init__(self, split):
        self.items=[]
        self.weights=[]
        self.train=split=="train"
        roots=["helmet-separate-v2-expanded-09"] + (["rain-front-01","rain-rider-02"] if self.train else [])
        for name in roots:
            root=Path("data/prepared")/name
            for line in (root/f"{split}-manifest.jsonl").read_text().splitlines():
                row=json.loads(line)
                for body in row["annotations"]:
                    if body["classId"]!=1:continue
                    rect=region(body["box"])
                    x,y,X,Y=rect
                    heads=[]
                    for a in row["annotations"]:
                        if a["classId"]<2:continue
                        l,t,r,b=a["box"]
                        if x<=(l+r)/2<X and y<=(t+b)/2<Y:
                            heads.append(([max(l,x),max(t,y),min(r,X),min(b,Y)],a["classId"]+1))
                    self.items.append((root/row["image"],rect,heads))
                    self.weights.append(5 if name.startswith("rain") else 1)
    def __len__(self):
        return len(self.items)
    def __getitem__(self,i):
        path,rect,heads=self.items[i]
        with Image.open(path) as im: crop=im.convert("RGB").crop(rect).resize((256,256),Image.Resampling.BILINEAR)
        x,y,X,Y=rect
        boxes=torch.tensor([[(l-x)*256/(X-x),(t-y)*256/(Y-y),(r-x)*256/(X-x),(b-y)*256/(Y-y)] for (l,t,r,b),_ in heads],dtype=torch.float32).reshape(-1,4)
        if self.train and random.random()<.5:
            crop=ImageOps.mirror(crop)
            boxes[:,[0,2]]=256-boxes[:,[2,0]]
        return pil_to_tensor(crop).float()/255,dict(boxes=boxes,labels=torch.tensor([c for _,c in heads],dtype=torch.int64))


def train():
    random.seed(20261125);torch.manual_seed(20261125)
    model,cfg,device,*_=load_model("joint-multiscale-02")
    cfg = dict(cfg, initialization="joint_context_transfer", parent_sha256=hashlib.sha256(Path("runs/joint-multiscale-02/best.pt").read_bytes()).hexdigest())
    folder=Path("runs/rider-crop-heads-01");folder.mkdir(exist_ok=False)
    tr=Crops("train");val=Crops("val")
    loader=DataLoader(tr,batch_size=4,sampler=WeightedRandomSampler(tr.weights,len(tr),True),collate_fn=collate_detection)
    validation=DataLoader(val,batch_size=2,collate_fn=collate_detection)
    opt=torch.optim.AdamW(model.parameters(),lr=.000025)
    best=-1;history=[]
    for epoch in range(1,7):
        model.train()
        for layer in model.modules():
            if isinstance(layer,torch.nn.BatchNorm2d):layer.eval()
        for images,targets in loader:
            losses=model([im.to(device) for im in images],[{k:v.to(device) for k,v in t.items()} for t in targets])
            opt.zero_grad(set_to_none=True);loss=sum(losses.values())
            if not torch.isfinite(loss):raise ValueError("Nonfinite loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        result=evaluate(model,validation)
        score=result["metrics"]["map50"]
        history.append(dict(epoch=epoch,validation=result))
        if score>best:
            best=score
            torch.save(dict(model=model.state_dict(),config=dict(cfg,run=folder.name,head_crop_only=True),epoch=epoch),folder/"best.pt")
        (folder/"history.json").write_text(json.dumps(history,indent=2))
        print(epoch,score,flush=True)


@torch.inference_mode()
def recover(image,objects,loaded):
    updated=list(objects)
    for rider in summarize(objects)["ocupantes"]:
        if rider["estado"]!="indeterminado":continue
        rect=region(rider["box"])
        crop=image.crop(rect).resize((256,256),Image.Resampling.BILINEAR)
        def candidates(flip):
            out=[]
            for p in detector_objects(*loaded[:3],ImageOps.mirror(crop) if flip else crop,.8):
                if p["class_id"]<2:continue
                l,t,r,b=p["box"]
                if flip:l,r=256-r,256-l
                x,y,X,Y=rect
                coords=[max(0,x+l*(X-x)/256),max(0,y+t*(Y-y)/256),min(image.width,x+r*(X-x)/256),min(image.height,y+b*(Y-y)/256)]
                if coords[0]<coords[2] and coords[1]<coords[3]:out.append(dict(p,box=coords,source="rider_crop_specialist"))
            return out
        first=candidates(False)
        if len(first)!=1:continue
        second=candidates(True)
        h=first[0]
        if len(second)!=1 or second[0]["class_id"]!=h["class_id"] or box_overlap(h["box"],second[0]["box"])<.5:continue
        if any(p["class_id"]>=2 and box_overlap(h["box"],p["box"])>.25 for p in updated):continue
        proposed=updated+[h]
        if summarize(proposed)["ocupantes"][rider["id"]-1]["estado"]!="indeterminado":updated=proposed
    return updated

if __name__=="__main__":train()
