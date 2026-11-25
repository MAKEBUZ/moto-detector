"""Evaluate a from-scratch crop specialist on actual detector candidates."""
import json
from pathlib import Path
import torch
from PIL import Image, ImageOps
from torchvision.ops import nms
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from data import SeparateHelmetDataset
from deterministic_gpu import configure_deterministic_gpu
from evaluate_reserved import counts_at_threshold
from metrics import detection_ap50
from models import HelmetCNN, build_detector

def classify_heads(image, detection, classifier):
    head_ids = torch.where(detection["labels"] >= 3)[0]
    heads = head_ids[nms(detection["boxes"][head_ids], detection["scores"][head_ids], .5)]
    others = torch.where(detection["labels"] < 3)[0]
    output = {k:v[others] for k,v in detection.items()}
    if not len(heads):
        return output
    crops = []
    photo = to_pil_image(image)
    for box in detection["boxes"][heads].tolist():
        left,top,right,bottom = box
        bounds = (int(left),int(top),max(int(left)+1,round(right)),max(int(top)+1,round(bottom)))
        crop = ImageOps.pad(photo.crop(bounds),(96,96),method=Image.Resampling.BILINEAR,color=(127,127,127))
        crops.append(pil_to_tensor(crop).float()/127.5-1)
    confidence, category = classifier(torch.stack(crops).cuda()).softmax(1).max(1)
    output["boxes"] = torch.cat((output["boxes"],detection["boxes"][heads]))
    output["scores"] = torch.cat((output["scores"],(confidence*detection["scores"][heads]).sqrt()))
    output["labels"] = torch.cat((output["labels"],category+3))
    return output

@torch.inference_mode()
def main():
    configure_deterministic_gpu()
    base = Path("reports/training/paired-70-73")
    ck = torch.load(json.loads((base/"selection.json").read_text())["checkpoint"],map_location="cpu",weights_only=True)
    c = ck["config"]
    model = build_detector(5,c["feature_stride"],c["proposal_multiplier"],deterministic_roi=True).cuda().eval()
    model.load_state_dict(ck["model"])
    specialist = HelmetCNN().cuda().eval()
    best = torch.load("runs/crop-specialist-01/best.pt",map_location="cpu",weights_only=True)
    assert best["config"]["pretrained"] is False
    specialist.load_state_dict(best["model"])
    ds = SeparateHelmetDataset(c["dataset"],"val",augment=False)
    samples = list(ds)
    targets = [t for _,t in samples]
    predictions = {"detector":[],"specialist":[]}
    for start in range(0,len(samples),2):
        images = [im for im,_ in samples[start:start+2]]
        for image,p in zip(images,model([im.cuda() for im in images])):
            for key,value in (("detector",p),("specialist",classify_heads(image,p,specialist))):
                predictions[key].append({k:v.cpu() for k,v in value.items()})
    report = {}
    for key,preds in predictions.items():
        report[key] = dict(metrics=detection_ap50(preds,targets,(1,2,3,4)),counts=counts_at_threshold(preds,targets,.5),
                           per_image={r["image_id"]:counts_at_threshold([p],[t],.5) for r,p,t in zip(ds.rows,preds,targets)})
    assert report["detector"]["counts"]==json.loads((base/"validation.json").read_text())["70"]["counts"]
    report["crop_validation"]=best["validation"]
    out=Path("reports/training/crop-specialist-01.json")
    out.write_text(json.dumps(report,indent=2))
    for k in predictions: print(k,report[k]["metrics"]["map50"],[(r["tp"],r["fp"],r["fn"]) for r in report[k]["counts"]])
    print(best["epoch"],best["validation"])
if __name__=="__main__":
    main()
