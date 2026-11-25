"""Trace actual validation head boxes before and after the crop specialist."""
import json
from pathlib import Path

import torch
from torchvision.ops import box_iou, nms
from collections import OrderedDict
from data import SeparateHelmetDataset
from deterministic_gpu import configure_deterministic_gpu
from evaluate_crop_specialist import classify_heads
from evaluate_reserved import counts_at_threshold
from models import HelmetCNN, build_detector


@torch.inference_mode()
def main():
    configure_deterministic_gpu()
    folder = Path("reports/training")
    ck = torch.load("runs/paired-n70-s20261124/last.pt", map_location="cpu", weights_only=True)
    config = ck["config"]
    detector = build_detector(5, config["feature_stride"], config["proposal_multiplier"], deterministic_roi=True).cuda().eval()
    detector.load_state_dict(ck["model"])
    specialist = HelmetCNN().cuda().eval()
    specialist.load_state_dict(torch.load("runs/crop-specialist-01/best.pt", map_location="cpu", weights_only=True)["model"])
    ds = SeparateHelmetDataset(config["dataset"], "val", augment=False)
    samples = list(ds)
    rows, predictions = [], []
    for start in range(0,len(ds),2):
        images = [x for x,_ in samples[start:start+2]]
        transformed, scaled = detector.transform([x.cuda() for x in images], [
            {k:v.cuda() for k,v in t.items()} for _,t in samples[start:start+2]])
        features = OrderedDict([("0", detector.backbone(transformed.tensors))])
        proposals, _ = detector.rpn(transformed, features)
        for i,(image,p) in enumerate(zip(images,detector([x.cuda() for x in images]))):
            target = samples[start+i][1]
            ids = torch.where(p["labels"]>=3)[0]
            kept = ids[nms(p["boxes"][ids],p["scores"][ids],.5)]
            result = classify_heads(image,p,specialist)
            predictions.append({k:v.cpu() for k,v in result.items()})
            head = result["labels"]>=3
            assert torch.equal(result["boxes"][head],p["boxes"][kept])
            for j in torch.where(target["labels"]==4)[0].tolist():
                box = target["boxes"][j:j+1].cuda()
                before = box_iou(box,p["boxes"][ids])[0]
                after = box_iou(box,p["boxes"][kept])[0]
                candidates = []
                for k in torch.where(after>=.5)[0].tolist():
                    score = float(result["scores"][head][k])
                    source_score = float(p["scores"][kept[k]])
                    candidates.append(dict(iou=float(after[k]),label=int(result["labels"][head][k]),combined_score=score,
                                           detector_score=source_score,specialist_confidence=score**2/source_score))
                reason = "detectada"
                if not any(c["label"]==4 and c["combined_score"]>=.5 for c in candidates):
                    if not (before>=.5).any(): reason="sin caja cercana entre las detecciones de cabeza"
                    elif not candidates: reason="perdida en NMS previa al especialista"
                    elif not any(c["label"]==4 for c in candidates): reason="clasificada con casco"
                    else: reason="puntuación combinada inferior a 0,5"
                rows.append(dict(image=ds.rows[start+i]["image_id"],index=j,rpn_max_iou=float(box_iou(scaled[i]["boxes"][j:j+1],proposals[i]).max()),max_iou_before=float(before.max()) if len(before) else 0,
                                 max_iou_after=float(after.max()) if len(after) else 0,candidates=candidates,reason=reason))
    counts=counts_at_threshold(predictions,[t for _,t in samples],.5)
    assert counts==json.loads((folder/"crop-specialist-01.json").read_text())["specialist"]["counts"]
    assert sum(r["reason"]=="detectada" for r in rows)==counts[3]["tp"]
    (folder/"crop-specialist-head-trace.json").write_text(json.dumps(rows,indent=2),encoding="utf8")
    for row in rows: print(row)
if __name__=="__main__":
    main()
