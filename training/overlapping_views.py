"""Recover distinct road users supported by overlapping views of the same frame."""
from PIL import Image
from photo_results import box_overlap, summarize


def recover_views(image, base, detect):
    w, h = image.size
    if min(w, h) < 32:
        return base
    if h > w:
        windows = [(0, top, w, min(h, top+w)) for top in range(0, h-w+1, max(1,w//2))]
        windows.append((0,h-w,w,h))
    else:
        windows = [(0,0,w,h)]
        for frac in (.67,.50):
            width, height = round(frac*w), round(frac*h)
            for x in (0,(w-width)//2,w-width):
                for y in ((h-height)//2,h-height):
                    windows.append((x,y,x+width,y+height))
    views = []
    for l,t,r,b in dict.fromkeys(windows):
        tile = image.crop((l,t,r,b))
        scale = min(2., 640/max(tile.size))
        tw,th = max(1,round(tile.width*scale)), max(1,round(tile.height*scale))
        tile = tile.resize((tw,th),Image.Resampling.BILINEAR)
        items = []
        for obj in detect(tile,.65):
            x,y,X,Y = obj["box"]
            if (l and x<2) or (t and y<2) or (r<w and X>tw-2) or (b<h and Y>th-2):
                continue
            box = [l+x*(r-l)/tw,t+y*(b-t)/th,l+X*(r-l)/tw,t+Y*(b-t)/th]
            items.append(dict(obj,box=box,source="overlapping_views"))
        views.append(items)
    merged = list(base)
    candidates = []
    for i,items in enumerate(views):
        for obj in items:
            if any(q["class_id"]==obj["class_id"] and box_overlap(q["box"],obj["box"])>=.55
                   for j,other in enumerate(views) if j!=i for q in other):
                candidates.append(obj)
    for obj in sorted(candidates,key=lambda o:-o["score"]):
        if not any((q["class_id"]==obj["class_id"] or min(q["class_id"],obj["class_id"])>=2)
                   and box_overlap(q["box"],obj["box"])>.25 for q in merged):
            merged.append(obj)
    # Admit additional boxes only in an unambiguously linked bike/body/head trio.
    scene=summarize(merged)
    bikes=[q for q in merged if q["class_id"]==0]
    people=[q for q in merged if q["class_id"]==1]
    keep={id(q) for q in base}
    for p in scene["ocupantes"]:
        if p["moto"] and p["estado"]!="indeterminado":
            keep.update([id(bikes[p["moto"]-1]),id(people[p["id"]-1]),id(p["cabezas"][0])])
    return [q for q in merged if id(q) in keep]
