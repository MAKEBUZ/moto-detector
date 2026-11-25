"""Learn four-class predictions consistent across image conditions on six training sources."""
import copy
import hashlib
import json
import random
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor
from adapt_all_photos import ProjectPhotos
from data import SeparateHelmetDataset, collate_detection
from models import build_detector, forbid_external_weights
from train_joint_clean import evaluate


class Photos(ProjectPhotos):
    def __getitem__(self,index):
        with Image.open(self.rows[index]["image"]) as im:
            im=ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((640,640),Image.Resampling.BILINEAR)
            return pil_to_tensor(im).float().div_(255)


def photo_batch(images):
    return images


def outputs(model, features, proposals, shapes):
    pooled=model.roi_heads.box_roi_pool(features,proposals,shapes)
    return model.roi_heads.box_predictor(model.roi_heads.box_head(pooled))[0]


def consistent_loss(model,teacher,images):
    with torch.no_grad():
        weak,_=teacher.transform(images,None)
        features=OrderedDict([("0",teacher.backbone(weak.tensors))])
        proposals,_=teacher.rpn(weak,features)
        boxes=[b[:96] for b in proposals]
        soft=outputs(teacher,features,boxes,weak.image_sizes).float().div(2).softmax(-1)
    strong=[((im-.5)*random.uniform(.7,1.3)+.5+random.uniform(-.15,.15)).clamp(0,1) for im in images]
    batch,_=model.transform(strong,None)
    assert batch.image_sizes==weak.image_sizes
    student=OrderedDict([("0",model.backbone(batch.tensors))])
    logits=outputs(model,student,boxes,batch.image_sizes)
    return 4*F.kl_div((logits.float()/2).log_softmax(-1),soft,reduction="batchmean")


def main():
    seed=20261124
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    forbid_external_weights()
    run=Path("runs/joint-six-sources-01")
    run.mkdir(exist_ok=False)
    parent=Path("runs/joint-clean640-01/best.pt")
    ck=torch.load(parent,map_location="cpu",weights_only=True)
    model=build_detector(5,8,4).cuda()
    model.load_state_dict(ck["model"])
    teacher=copy.deepcopy(model).eval()
    average=copy.deepcopy(teacher)
    data=Photos(run)
    val=SeparateHelmetDataset(ck["config"]["dataset"],"val",augment=False)
    reviewed=SeparateHelmetDataset(ck["config"]["dataset"],"train",photometric=True)
    assert len(data)==7222 and not {r["groupId"] for r in data.rows}&{r["group_id"] for r in val.rows}
    photos=DataLoader(data,batch_size=8,num_workers=3,persistent_workers=True,pin_memory=True,shuffle=True,collate_fn=photo_batch)
    labeled=DataLoader(reviewed,batch_size=3,num_workers=2,persistent_workers=True,shuffle=True,collate_fn=collate_detection)
    validation=DataLoader(val,batch_size=2,collate_fn=collate_detection)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.000012,weight_decay=.0001)
    scaler=torch.amp.GradScaler("cuda")
    batchiter=iter(labeled)
    best=-1
    updates=0
    history=[]
    for epoch in range(1,4):
        model.train()
        for mod in model.modules():
            if isinstance(mod,torch.nn.BatchNorm2d): mod.eval()
        total=0.
        for index, images in enumerate(photos):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss=consistent_loss(model,teacher,[im.cuda(non_blocking=True) for im in images])
                if index%3==0:
                    try: ims,targets=next(batchiter)
                    except StopIteration:
                        batchiter=iter(labeled)
                        ims,targets=next(batchiter)
                    losses=model([im.cuda() for im in ims],[{k:v.cuda() for k,v in t.items()} for t in targets])
                    loss=loss+sum(losses.values())
            if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                for a,p in zip(average.parameters(),model.parameters()): a.lerp_(p,.002)
            total+=loss.item()
            updates+=1
        result=evaluate(average,validation)
        score=result["metrics"]["map50"]
        history.append(dict(epoch=epoch,updates=updates,loss=total/len(photos),**result))
        (run/"history.json").write_text(json.dumps(history,indent=2))
        if score>best:
            best=score
            cfg=dict(ck["config"],run=run.name,initialization="project_consistency",external_pretrained=False,
                     parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),consistency_images=len(data))
            torch.save(dict(model=average.state_dict(),epoch=epoch,config=cfg,optimizer_updates=updates,validation=result["metrics"]),run/"best.pt")
        print(epoch,updates,score,flush=True)


if __name__=="__main__":
    main()
