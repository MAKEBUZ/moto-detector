"""Compare the two final checkpoints and evaluate only the prespecified winner on reserve 02."""
import hashlib
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

OUT = Path("reports/training/paired-70-73")
def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2), encoding="utf8")

def evaluate(model, samples, names):
    predictions = []
    for start in range(0,len(samples),2):
        images = [image.cuda() for image,_ in samples[start:start+2]]
        predictions.extend([{k:v.cpu() for k,v in p.items()} for p in model(images)])
    targets = [target for _,target in samples]
    return dict(metrics=detection_ap50(predictions,targets,(1,2,3,4)),
                counts=counts_at_threshold(predictions,targets,.5),
                images={name:counts_at_threshold([p],[t],.5) for name,p,t in zip(names,predictions,targets)})

def reserve(image_size=640):
    root = Path("data/prepared/helmet-separate-v2-test-02")
    rows = [json.loads(line) for line in (root/"test-manifest.jsonl").read_text().splitlines()]
    samples = []
    assert len(rows)==6
    for row in rows:
        path=Path(row["image"])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row["sha256"] and row["split"]=="test"
        with Image.open(path) as im:
            w,h=im.size
            scale=min(1,image_size/max(w,h))
            size=max(1,round(w*scale)),max(1,round(h*scale))
            image=pil_to_tensor(im.convert("RGB").resize(size,Image.Resampling.BILINEAR)).float()/255
        boxes=torch.tensor([a["box"] for a in row["annotations"]])*torch.tensor([size[0]/w,size[1]/h]*2)
        samples.append((image,dict(boxes=boxes,labels=torch.tensor([a["classId"]+1 for a in row["annotations"]]))))
    return samples,[row["image_id"] for row in rows]

@torch.inference_mode()
def main():
    configure_deterministic_gpu()
    reports={}; models={}; configs=[]
    for n in (70,73):
        folder=Path(f"runs/paired-n{n}-s20261124")
        ck=torch.load(folder/"last.pt",map_location="cpu",weights_only=True)
        c=ck["config"];configs.append(c)
        assert ck["optimizer_updates"]==1950 and c["deterministic_gpu"] and not c["pretrained"]
        model=build_detector(5,c["feature_stride"],c["proposal_multiplier"],deterministic_roi=True).cuda().eval()
        model.load_state_dict(ck["model"]);models[n]=model
        ds=SeparateHelmetDataset(c["dataset"],"val",augment=False)
        reports[n]=evaluate(model,list(ds),[r["image_id"] for r in ds.rows])
        assert abs(reports[n]["metrics"]["map50"]-ck["validation"]["map50"])<1e-6
    assert configs[0]["initial_state_sha256"]==configs[1]["initial_state_sha256"]
    winner=max(reports,key=lambda n:reports[n]["metrics"]["map50"])
    save("validation.json",reports)
    path=Path(f"runs/paired-n{winner}-s20261124/last.pt")
    save("selection.json",dict(winner=winner,checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),threshold=.5))
    save("reserve.json",evaluate(models[winner],*reserve()))
    print("winner",winner)
if __name__=="__main__":
    main()
