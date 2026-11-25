"""Confirm an unresolved head near the image edge using two padded views."""
from math import ceil
from PIL import ImageOps
from photo_results import summarize


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0.


def recover(image, predictions, detect, threshold):
    w, h = image.size
    if max(w, h) > 352:
        return []
    people = summarize(predictions)["ocupantes"]
    eligible = []
    for p in people:
        l,t,r,_ = p["box"]
        if p["estado"] == "indeterminado" and not p["cabezas"] and min(l/w,t/h,(w-r)/w) < .12:
            eligible.append(p["id"])
    if not eligible:
        return []
    views = []
    for fraction in (.1,.2):
        pad = 8 * ceil(max(w,h)*fraction/8)
        proposals = []
        for obj in detect(ImageOps.expand(image,border=pad,fill=(127,127,127)),max(.8,threshold)):
            if obj["class_id"] < 2:
                continue
            x,y,X,Y = obj["box"]
            box = [max(0,x-pad),max(0,y-pad),min(w,X-pad),min(h,Y-pad)]
            if box[0]<box[2] and box[1]<box[3]:
                proposals.append(dict(obj,box=box))
        views.append(summarize(predictions+proposals)["ocupantes"])
    accepted = []
    for key in eligible:
        a,b = (view[key-1] for view in views)
        if a["estado"]!="indeterminado" and a["estado"]==b["estado"]:
            first,second = a["cabezas"][0],b["cabezas"][0]
            if iou(first["box"],second["box"])>=.5:
                accepted.append(dict(first,score=min(first["score"],second["score"]),source="padded_consensus"))
    return accepted


def recover_single_rider(image, predictions, detect, threshold):
    """Verify a missed head in two upper-body crops of an isolated rider."""
    people = [p for p in predictions if p["class_id"]==1]
    if len(people)!=1 or any(p["class_id"]>=2 for p in predictions):
        return []
    l,t,r,b = people[0]["box"]
    w,h = r-l,b-t
    agreed = []
    for margin in (.15,.20):
        x,y = max(0,int(l-margin*w)),max(0,int(t-.25*h))
        right,bottom = min(image.width,int(r+margin*w)),min(image.height,int(t+.55*h))
        if right<=x or bottom<=y:
            return []
        candidates = []
        for p in detect(image.crop((x,y,right,bottom)),max(.5,threshold)):
            if p["class_id"]<2:
                continue
            L,T,R,B = p["box"]
            candidate = dict(p,box=[L+x,T+y,R+x,B+y])
            if summarize(people+[candidate])["ocupantes"][0]["estado"]!="indeterminado":
                candidates.append(candidate)
        if len(candidates)!=1:
            return []
        agreed.append(candidates[0])
    a,b = agreed
    if a["class_id"]!=b["class_id"] or max(a["score"],b["score"])<.7 or iou(a["box"],b["box"])<.5:
        return []
    return [dict(a,score=min(a["score"],b["score"]),source="upper_body_consensus")]
