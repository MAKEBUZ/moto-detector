"""Recheck a crowded single motorcycle when head and body counts disagree."""
from PIL import ImageOps
from photo_results import summarize, box_overlap


def reconsider(image, objects, detect, threshold):
    bikes = [p for p in objects if p["class_id"] == 0]
    bodies = [p for p in objects if p["class_id"] == 1]
    heads = [p for p in objects if p["class_id"] >= 2 and p["score"] >= .9]
    if len(bikes) != 1 or not 2 <= len(heads) <= 5 or not bodies or len(bodies) == len(heads):
        return objects
    x,y,X,Y = bikes[0]["box"]
    width = X-x
    for p in heads:
        l,t,r,b = p["box"]
        if not (x-.15*width <= (l+r)/2 <= X+.15*width and b < y+.5*(Y-y)):
            return objects
    mirror = detect(ImageOps.mirror(image), max(.5,threshold))
    restored = []
    for p in mirror:
        l,t,r,b = p["box"]
        restored.append(dict(p,box=[image.width-r,t,image.width-l,b]))
    for head in heads:
        if not any(q["class_id"]==head["class_id"] and q["score"]>=.8 and
                   box_overlap(q["box"],head["box"])>=.55 for q in restored):
            return objects
    # Preserve distinct lower-confidence heads unless covered by the confirmed group.
    for p in objects:
        if p["class_id"]>=2 and p not in heads:
            l,t,r,b=p["box"]
            if not any(box_overlap(p["box"],h["box"])>.4 or
                       (l<= (h["box"][0]+h["box"][2])/2 <=r and t<= (h["box"][1]+h["box"][3])/2 <=b)
                       for h in heads):
                return objects
    if len(bodies)>len(heads):
        candidates=[q for q in restored if q["class_id"]==1]
    else:
        head_w=max(h["box"][2]-h["box"][0] for h in heads)
        left=max(0,int(min(h["box"][0] for h in heads)-2*head_w))
        right=min(image.width,int(max(h["box"][2] for h in heads)+2.5*head_w))
        top=max(0,int(min(h["box"][1] for h in heads)-head_w))
        bottom=min(image.height,int(y+.72*(Y-y)))
        if right<=left or bottom<=top:
            return objects
        candidates=[]
        for q in detect(image.crop((left,top,right,bottom)),max(.5,threshold)):
            if q["class_id"]==1:
                l,t,r,b=q["box"]
                candidates.append(dict(q,box=[l+left,t+top,r+left,b+top]))
    if len(candidates)!=len(heads):
        return objects
    if not all(any(box_overlap(p["box"],q["box"])>=.25 for q in candidates) for p in bodies):
        return objects
    proposed=bikes+[dict(q,source="crowded_view") for q in candidates]+heads
    riders=summarize(proposed)["ocupantes"]
    if all(p["moto"] and p["estado"]!="indeterminado" for p in riders):
        return proposed
    return objects

def recover_hidden_people(image, objects, detect, threshold):
    """Seek independent body evidence for distinct heads above one bike."""
    from itertools import permutations
    from math import hypot
    bikes = sorted((p for p in objects if p["class_id"]==0),key=lambda p:-p["score"])
    people = [p for p in objects if p["class_id"]==1]
    if not 1<=len(bikes)<=2 or len(people)!=1:
        return objects
    x,y,X,Y=bikes[0]["box"]
    bw,bh=X-x,Y-y
    if len(bikes)==2:
        a,b,A,B=bikes[1]["box"]
        inside=max(0,min(X,A)-max(x,a))*max(0,min(Y,B)-max(y,b))
        if inside < .95*(A-a)*(B-b) or (A-a)*(B-b)>.75*bw*bh:
            return objects
    heads=[]
    for q in sorted(objects,key=lambda p:-p["score"]):
        if q["class_id"]<2: continue
        l,t,r,b=q["box"]
        if not x <= (l+r)/2 <= X or b >= y+.5*bh: continue
        if not any(box_overlap(q["box"],k["box"])>.35 for k in heads):
            heads.append(q)
    if not 2<=len(heads)<=4:
        return objects

    def view(rect, flip=False):
        l,t,r,b=rect
        crop=image.crop(rect)
        found=detect(ImageOps.mirror(crop) if flip else crop,threshold)
        restored=[]
        for q in found:
            a,v,A,V=q["box"]
            if flip: a,A=crop.width-A,crop.width-a
            restored.append(dict(q,box=[a+l,v+t,A+l,V+t]))
        return restored

    mirror=view((0,0,image.width,image.height),True)
    if len(bikes)==2 and not any(q["class_id"]==0 and q["score"]>=.9 and box_overlap(q["box"],bikes[0]["box"])>.55 for q in mirror):
        return objects
    upper=view((max(0,int(x-.25*bw)),max(0,int(y-.8*bh)),min(image.width,int(X+.25*bw)),min(image.height,int(y+.4*bh))))
    candidates=people+[q for q in mirror if q["class_id"]==1]
    confirmed=[]
    tentative=[]
    for head in heads:
        l,t,r,b=head["box"];w,h=r-l,b-t
        def evidence(rows):
            matches=[q for q in rows if q["class_id"]>=2 and box_overlap(q["box"],head["box"])>=.45]
            return max(matches,key=lambda q:q["score"],default=None)
        rev=evidence(mirror)
        focused=evidence(upper)
        if not rev and not focused: continue
        reliable=rev and (rev["class_id"]==head["class_id"] or (focused and focused["class_id"]==rev["class_id"]))
        chosen=dict(head)
        if reliable:
            chosen.update(class_id=rev["class_id"], **{"class":rev["class"]})
            confirmed.append(chosen)
        else:
            tentative.append(chosen)
        rect=(max(0,int(l-1.2*w)),max(0,int(t-.6*h)),min(image.width,int(r+1.2*w)),min(image.height,int(b+3*h)))
        candidates.extend(q for q in view(rect) if q["class_id"]==1)
    anchors=confirmed+tentative
    if len(anchors)<2 or len(candidates)>18: return objects
    # One detected body per independently located head; reject duplicate torso boxes.
    options=[]
    for q in anchors:
        l,t,r,b=q["box"];cx,cy=(l+r)/2,(t+b)/2
        fitting=[]
        for i,p in enumerate(candidates):
            a,v,A,V=p["box"];w,h=A-a,V-v
            if a<=cx<=A and v-.08*h<=cy<=v+.55*h and x-.1*bw<=(a+A)/2<=X+.1*bw and V>=y:
                cost=hypot((cx-(a+A)/2)/w,(cy-v)/h-.16)-.12*p["score"]
                fitting.append((cost,i))
        options.append(sorted(fitting)[:4])
    if any(not options[i] for i in range(len(confirmed))):
        return objects
    options = [group for group in options if group]
    if len(options)<2:
        return objects
    from itertools import product
    best=None
    for choice in product(*options):
        ids=[i for _,i in choice]
        if len(set(ids))<len(ids) or any(box_overlap(candidates[i]["box"],candidates[j]["box"])>.75 for i,j in permutations(ids,2)):
            continue
        cost=sum(v for v,_ in choice)
        if best is None or cost<best[0]:best=(cost,ids)
    if best is None:return objects
    result=[bikes[0]]+[dict(candidates[i],source="head_guided_view") for i in best[1]]+confirmed
    summary=summarize(result)
    if sum(p["estado"]!="indeterminado" for p in summary["ocupantes"])!=len(confirmed):
        return objects
    return result
