"""Recover an unresolved head from a rider crop and its horizontal reflection."""
from PIL import ImageOps
from photo_results import summarize, box_overlap


def add_heads(image, objects, detect):
    result = list(objects)
    for rider in summarize(objects)["ocupantes"]:
        if rider["estado"] != "indeterminado":
            continue
        l,t,r,b = rider["box"]
        cx = (l+r)/2
        height = b-t
        radius = max(height, r-l)
        cy = t+.20*height
        x,y = max(0,int(cx-radius)), max(0,int(cy-radius))
        crop = image.crop((x,y,min(image.width,int(cx+radius)),min(image.height,int(cy+radius))))
        def heads(mirror):
            candidates = []
            for p in detect(ImageOps.mirror(crop) if mirror else crop):
                if p["class_id"] < 2:
                    continue
                L,T,R,B = p["box"]
                if mirror:
                    L,R = crop.width-R,crop.width-L
                box = [L+x,T+y,R+x,B+y]
                if l <= (box[0]+box[2])/2 <= r and t-.08*height <= (box[1]+box[3])/2 <= t+.32*height:
                    candidates.append(dict(p,box=box))
            return candidates
        first = heads(False)
        if len(first) != 1:
            continue
        second = heads(True)
        p = first[0]
        if len(second)!=1 or second[0]["class_id"]!=p["class_id"] or box_overlap(p["box"],second[0]["box"])<.5:
            continue
        if any(q["class_id"]>=2 and box_overlap(q["box"],p["box"])>.25 for q in result):
            continue
        new = result+[p]
        if summarize(new)["ocupantes"][rider["id"]-1]["estado"] != "indeterminado":
            result = new
    return result
