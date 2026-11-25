"""Additional native-resolution views for wide, high-resolution videos."""
from PIL import Image
from photo_results import box_overlap, summarize


def recover_wide(image, base, detect):
    w,h=image.size
    if w < 1920 or w/h < 1.45:
        return base
    side=round(w*.60)
    tile_height=min(h,round(h*.68))
    riders=summarize(base)["ocupantes"]
    targets=[p for p in riders if p["estado"]=="indeterminado"]
    views=set()
    for person in targets:
        l,t,r,b=person["box"]
        cx,cy=(l+r)/2,t+.18*(b-t)
        views.add((0 if cx<w/2 else w-side, 0 if cy<h/2 else h-tile_height))
    candidates=[]
    for x,y in sorted(views):
        crop=image.crop((x,y,x+side,y+tile_height))
        for obj in detect(crop,.8):
            if obj["class_id"] != 2:
                continue
            l,t,r,b=obj["box"]
            if ((x and l<4) or (y and t<4) or
                (x+side<w and r>side-4) or (y+tile_height<h and b>tile_height-4)):
                continue
            candidates.append(dict(obj,box=[l+x,t+y,r+x,b+y],source="wide_video_tile"))
    result=list(base)
    for p in sorted(candidates,key=lambda o:-o["score"]):
        if any(q["class_id"]>=2 and box_overlap(p["box"],q["box"])>=.30 for q in result):
            continue
        trial=result+[p]
        after=summarize(trial)["ocupantes"]
        if any(before["estado"] != "indeterminado" and before["cabezas"] != now["cabezas"]
               for before,now in zip(riders,after)):
            continue
        if any(now["cabezas"]==[p] for now in after):
            result=trial
            riders=after
    return result


class WideVideoHeads:
    """Check wide-video detections using local models trained on head crops."""
    def __init__(self):
        import torch
        from pathlib import Path
        from models import HelmetCNN
        from predict import load_model
        self.head_model = load_model("rider-crop-heads-01")
        self.device = self.head_model[2]
        path = Path(__file__).resolve().parents[1]/"runs/crop-specialist-01/best.pt"
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        cfg = checkpoint["config"]
        if cfg["initialization"] != "random" or cfg["pretrained"] is not False:
            raise ValueError("Se esperaba el clasificador local entrenado desde cero.")
        self.crop_model = HelmetCNN()
        self.crop_model.load_state_dict(checkpoint["model"])
        self.crop_model.to(self.device).eval()

    def refine(self, image, base, detect):
        import torch
        from PIL import ImageOps
        from torchvision.transforms.functional import pil_to_tensor
        from rider_crop_heads import recover
        expanded = recover_wide(image, base, detect)
        candidates = recover(image, expanded, self.head_model)
        # A cropped torso at the upper edge can look like a complete rider to the
        # detector. Avoid treating a patch of jacket or windshield as its head.
        interior_heads = {id(head) for rider in summarize(candidates)["ocupantes"]
                          if rider["box"][1] > .02 * image.height for head in rider["cabezas"]}
        result = []
        for obj in candidates:
            category = obj["class_id"]
            new_candidate = obj.get("source") in ("wide_video_tile", "rider_crop_specialist")
            if new_candidate and id(obj) not in interior_heads:
                continue
            check_helmet = category == 2 and id(obj) in interior_heads and obj["score"] >= .85
            if category != 3 and not new_candidate and not check_helmet:
                result.append(obj)
                continue
            crop = ImageOps.pad(image.crop(tuple(obj["box"])), (96,96),
                                method=Image.Resampling.BILINEAR, color=(127,127,127))
            x = pil_to_tensor(crop).to(self.device).float()/127.5-1
            with torch.inference_mode():
                probabilities = self.crop_model(torch.stack((x,x.flip(-1)))).softmax(1)
            confidence = probabilities[:,category-2].min().item()
            if (not new_candidate and category == 2) or confidence >= .8:
                result.append(dict(obj, verified_helmet=True) if check_helmet and confidence >= .9 else obj)
        return result
