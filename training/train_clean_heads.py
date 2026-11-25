"""Continue the project's three-class detector at 960px on all clean training scenes."""
import json
import math
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from data import HelmetDataset, collate_detection, read_jsonl
from evaluate_reserved import counts_at_threshold
from models import build_detector, forbid_external_weights


def dataset(split):
    ds = HelmetDataset("data/prepared/helmet-baseline-v1","detector",split)
    clean = Path("EdgeVision_v1_YOLO_Limpio")
    paths = {p.stem:p for p in (clean/"images").glob("*/*.jpg")}
    if split == "train":
        existing = {Path(r["image"]).stem for r in ds.rows}
        for r in read_jsonl("reports/preparation/image-decisions.jsonl"):
            if r["source"] != "edgevision" or r["split"] != "train" or Path(r["image"]).stem in existing:
                continue
            if "detector_conflicting_head_labels" not in r["reasons"]:
                ds.rows.append(dict(image=r["image"],groupId=r["groupId"]))
    for row in ds.rows:
        image = paths[Path(row["image"]).stem]
        with Image.open(image) as im:
            w,h = im.size
        labels = []
        for line in (clean/"labels"/image.parent.name/(image.stem+".txt")).read_text().splitlines():
            cls,*coords = line.split()
            x,y,bw,bh = map(float,coords)
            labels.append(dict(classId=int(cls),box=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
        row.update(image=str(image),annotations=labels)
    ds.root,ds.separate,ds.image_size = Path("."),True,960
    return ds


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    totals = {k:0 for k in ("tp","fp","fn")}
    for images,targets in loader:
        preds = model([x.cuda() for x in images])
        counts = counts_at_threshold([{k:v.cpu() for k,v in p.items()} for p in preds],targets)[1]
        for k in totals:
            totals[k] += counts[k]
    tp,fp,fn = (totals[k] for k in ("tp","fp","fn"))
    return dict(totals,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn))


def main():
    run = Path("runs/clean-heads960-12")
    run.mkdir(exist_ok=False)
    seed = 20261124
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    forbid_external_weights()
    train,val = dataset("train"),dataset("val")
    assert len(train)>=1583 and len(val)==347
    assert not {r["groupId"] for r in train.rows} & {r["groupId"] for r in val.rows}
    (run/"dataset.json").write_text(json.dumps(dict(training=len(train),validation=len(val))))
    trainloader = DataLoader(train,batch_size=4,shuffle=True,num_workers=3,persistent_workers=True,pin_memory=True,collate_fn=collate_detection)
    valloader = DataLoader(val,batch_size=2,num_workers=2,persistent_workers=True,collate_fn=collate_detection)
    model = build_detector(4,8,4,image_size=960).cuda()
    source = torch.load("runs/edge-clean-20261124/transfer.pt",weights_only=True,map_location="cpu")
    assert source["external_weights"] is False and source["initialization"]=="random"
    model.load_state_dict(source["model"])
    from adapt_all_photos import adapt
    adapt(model,run)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=12,eta_min=.00002)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    best = -1
    for epoch in range(1,13):
        model.train()
        total = 0.
        for images,targets in trainloader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                parts=model([x.cuda(non_blocking=True) for x in images],
                            [{k:v.cuda(non_blocking=True) for k,v in t.items()} for t in targets])
                loss=sum(parts.values())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            scaler.step(optimizer)
            scaler.update()
            total+=loss.item()
        scheduler.step()
        record=dict(epoch=epoch,loss=total/len(trainloader))
        if epoch%3==0 or epoch==1:
            record.update(evaluate(model,valloader))
            if record["f1"]>best:
                best=record["f1"]
                torch.save(dict(model=model.state_dict(),epoch=epoch,classes=source["classes"],initialization="random",images=len(train),image_size=960,external_weights=False),run/"best.pt")
        history.append(record)
        (run/"history.json").write_text(json.dumps(history,indent=2))
        print(record,flush=True)
    model.load_state_dict(torch.load(run/"best.pt",weights_only=True)["model"])
    test=dataset("test")
    assert not {r["groupId"] for r in test.rows} & {r["groupId"] for r in train.rows+val.rows}
    result=dict(images=len(test),groups=len({r["groupId"] for r in test.rows}),**evaluate(model,DataLoader(test,batch_size=2,num_workers=2,collate_fn=collate_detection)))
    (run/"test.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)


if __name__=="__main__":
    main()
