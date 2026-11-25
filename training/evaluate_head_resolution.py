"""Paired validation of native-image inference at 1x and 1.5x resolution."""
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from data import SeparateHelmetDataset
from deterministic_gpu import configure_deterministic_gpu
from evaluate_reserved import counts_at_threshold
from metrics import detection_ap50
from models import build_detector


@torch.inference_mode()
def main():
    configure_deterministic_gpu()
    root = Path("data/prepared/helmet-separate-v2-expanded-07")
    folder = Path("reports/training/paired-70-73")
    selection = json.loads((folder / "selection.json").read_text())
    checkpoint = torch.load(selection["checkpoint"], map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    assert not config["pretrained"] and config["deterministic_gpu"]
    model = build_detector(5, config["feature_stride"], config["proposal_multiplier"], deterministic_roi=True)
    model.load_state_dict(checkpoint["model"])
    model.cuda().eval()
    ds = SeparateHelmetDataset(root, "val", augment=False)
    records = {}
    predictions_by_cap = {}
    sizes_by_cap = {}
    for cap in (640, 960):
        model.transform.min_size = (384 * cap // 640,)
        model.transform.max_size = cap
        samples = []
        for i,row in enumerate(ds.rows):
            if cap == 640:
                samples.append(ds[i])
                continue
            with Image.open(root / row["image"]) as original:
                w,h = original.size
                scale = min(1.,cap/max(w,h))
                nw,nh = max(1,round(w*scale)),max(1,round(h*scale))
                image = pil_to_tensor(original.convert("RGB").resize((nw,nh), Image.Resampling.BILINEAR)).float()/255
            boxes = torch.tensor([a["box"] for a in row["annotations"]])*torch.tensor([nw/w,nh/h]*2)
            labels = torch.tensor([a["classId"]+1 for a in row["annotations"]])
            samples.append((image,dict(boxes=boxes,labels=labels)))
        preds = []
        for start in range(0,len(samples),2):
            preds.extend([{k:v.cpu() for k,v in p.items()} for p in model([im.cuda() for im,_ in samples[start:start+2]])])
        predictions_by_cap[cap] = preds
        sizes_by_cap[cap] = [im.shape[-2:] for im,_ in samples]
        targets = [target for _,target in samples]
        metrics = detection_ap50(preds,targets,(1,2,3,4))
        counts = counts_at_threshold(preds,targets,.5)
        records[cap] = dict(metrics=metrics,counts=counts,images={r["image_id"]:counts_at_threshold([p],[t],.5) for r,p,t in zip(ds.rows,preds,targets)})
        print(cap,metrics["map50"],[(r["tp"],r["fp"],r["fn"]) for r in counts])
    # Use the larger view for helmet boxes only, mapped to reference coordinates.
    combined = []
    for low,high,(h,w),(H,W) in zip(predictions_by_cap[640],predictions_by_cap[960],sizes_by_cap[640],sizes_by_cap[960]):
        high = dict(high,boxes=high["boxes"]*torch.tensor([w/W,h/H]*2))
        combined.append({k:torch.cat([low[k][low["labels"]!=3],high[k][high["labels"]==3]]) for k in low})
    targets = [ds[i][1] for i in range(len(ds))]
    records["dual"] = dict(metrics=detection_ap50(combined,targets,(1,2,3,4)),counts=counts_at_threshold(combined,targets,.5))
    print("dual",records["dual"]["metrics"]["map50"],records["dual"]["counts"])
    previous = json.loads((folder/"validation.json").read_text())["70"]
    assert records[640]["counts"] == previous["counts"] and abs(records[640]["metrics"]["map50"]-previous["metrics"]["map50"])<1e-6
    (folder/"resolution-validation.json").write_text(json.dumps(records,indent=2))

if __name__ == "__main__":
    main()
