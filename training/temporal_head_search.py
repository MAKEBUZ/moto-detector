"""Match verified helmet patches across a short temporal gap in a wide video."""
from collections import defaultdict
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.transforms.functional import pil_to_tensor
from photo_results import box_overlap


def nearby_matches(gray, body, seeds, current, factor):
    height,width=gray.shape
    l,t,r,b=(v*factor for v in body);w,h=r-l,b-t
    candidates=[]
    for n,box,patch in seeds:
        gap=abs(n-current)
        if not 0<gap<=15:continue
        x,y,X,Y=box;bw,bh=X-x,Y-y
        if min(bw,bh)<7:continue
        # Use context around the helmet to disambiguate visually similar riders.
        cx,cy=(x+X)/2,(y+Y)/2
        search_w=max(18,gap*4+.3*bw);search_h=max(14,gap*4+.3*bh)
        ph,pw=patch.shape
        L=max(0,int(max(l-.12*w,cx-search_w)-pw/2));R=min(width,int(min(r+.12*w,cx+search_w)+pw/2))
        T=max(0,int(max(t-.32*h,cy-search_h)-ph/2));B=min(height,int(min(t+.52*h,cy+search_h)+ph/2))
        if R-L<pw or B-T<ph:continue
        values=cv2.matchTemplate(gray[T:B,L:R],patch,cv2.TM_CCOEFF_NORMED)
        _,score,_,loc=cv2.minMaxLoc(values)
        if score<.88:continue
        mx=L+loc[0]+pw/2;my=T+loc[1]+ph/2
        if not (l-.10*w<=mx<=r+.10*w and t-.32*h<=my<=t+.52*h):continue
        candidates.append(dict(seed=n,similarity=score,box=[(mx-bw/2)/factor,(my-bh/2)/factor,(mx+bw/2)/factor,(my+bh/2)/factor]))
    return candidates


def verify(image,box,verifier):
    crop=ImageOps.pad(image.crop(tuple(box)),(96,96),method=Image.Resampling.BILINEAR,color=(127,127,127))
    x=pil_to_tensor(crop).float().div(127.5).sub(1).to(verifier.device)
    with torch.inference_mode():
        return verifier.crop_model(torch.stack((x,x.flip(-1)))).softmax(1)[:,0].min().item()


def seed_patch(gray, box, factor):
    x,y,X,Y=(v*factor for v in box);cx,cy=(x+X)/2,(y+Y)/2
    rw=.85*(X-x);rh=.85*(Y-y)
    if min(X-x,Y-y)<7 or cx-rw<0 or cy-rh<0 or cx+rw>=gray.shape[1] or cy+rh>=gray.shape[0]:return None
    patch=gray[round(cy-rh):round(cy+rh),round(cx-rw):round(cx+rw)].copy()
    if patch.std()<9:return None
    return (x,y,X,Y),patch


class LiveHelmetSearch:
    """Recheck the current pixels using recent detector observations of the same track."""

    def __init__(self, verifier):
        self.verifier = verifier
        self.seeds = {}

    def refine(self, image, people, ids, tracker, frame):
        self.seeds = {key: [s for s in seeds if frame-s[0] <= 15]
                      for key, seeds in self.seeds.items() if seeds and frame-seeds[-1][0] <= 15}
        factor = min(1., 960/image.width)
        gray = np.asarray(image.convert("L").resize(
            (round(image.width*factor), round(image.height*factor)), Image.Resampling.BILINEAR))
        occupied = [h["box"] for p in people for h in p["cabezas"]]
        for person in people:
            key = ids.get(person["id"])
            if key is None:
                continue
            if person["cabezas"]:
                head = person["cabezas"][0]
                if person["estado"] == "con_casco" and head.get("verified_helmet"):
                    patch = seed_patch(gray, head["box"], factor)
                    if patch is not None:
                        self.seeds.setdefault(key, []).append((frame, *patch))
                elif person["estado"] == "sin_casco":
                    self.seeds.pop(key, None)
                continue
            if not tracker.confirmed(key) or key not in self.seeds:
                continue
            matches = nearby_matches(gray, person["box"], self.seeds[key], frame, factor)
            if not matches:
                continue
            best = max(matches, key=lambda m: m["similarity"])
            box = best["box"]
            if (any(box_overlap(box, other) > .25 for other in occupied)
                    or any(box_overlap(box, m["box"]) < .4 for m in matches)):
                continue
            confidence = verify(image, box, self.verifier)
            if confidence < .9:
                continue
            person["cabezas"] = [dict(box=box, class_id=2, score=confidence,
                                      source="temporal_patch_check", seed_frame=best["seed"])]
            person["estado"] = "con_casco"
            occupied.append(box)
            track = tracker.items[key]
            track["patch_checks"] = track.get("patch_checks", 0) + 1


def main():
    from audit_zero_heads import OUT,PREVIOUS
    from traffic_video import Tracks
    from photo_results import summarize
    from wide_video import WideVideoHeads
    from PIL import ImageDraw
    control=json.loads(PREVIOUS.read_text()); targets={r['id'] for r in control['trayectorias_ocupantes'] if not r['votos']}
    tr=Tracks();requests=defaultdict(list);refs=defaultdict(list)
    for line in (OUT/"baseline.jsonl").read_text().splitlines():
        row=json.loads(line);n=row['frame'];rs=summarize(row['objects'])['ocupantes'];keys=tr.update(rs,n)
        for p in rs:
            for head in p['cabezas']:
                if head.get('verified_helmet'):refs[n].append(head['box'])
            if keys[p['id']] in targets:
                other=[q for q in rs if q is not p]
                requests[n].append((keys[p['id']],p['box'],[h['box'] for q in other for h in q['cabezas']],[q['box'] for q in other]))
    needed={k for n in requests for k in range(max(1,n-15),min(control['frames'],n+15)+1)}
    cap=cv2.VideoCapture(control['source']);seeds=defaultdict(list);grays={};color={}
    # Decode sequentially; hold only requested images in JPEG form.
    for n in range(1,control['frames']+1):
        assert cap.grab()
        if n not in needed:continue
        ok,f=cap.retrieve();assert ok
        factor=min(1,960/f.shape[1])
        g=cv2.cvtColor(cv2.resize(f,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
        if n in requests:
            grays[n]=g;color[n]=cv2.imencode('.jpg',f,[cv2.IMWRITE_JPEG_QUALITY,96])[1]
        for box in refs[n]:
            pair=seed_patch(g,box,factor)
            if pair is not None:seeds[n].append((n,*pair))
    cap.release()
    verifier=WideVideoHeads()
    evidence=defaultdict(list);tiles=defaultdict(list)
    for n,items in requests.items():
        local=[s for k in range(n-15,n+16) for s in seeds[k]]
        im=Image.fromarray(cv2.cvtColor(cv2.imdecode(color[n],cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB))
        for key,body,used,others in items:
            matches=nearby_matches(grays[n],body,local,n,factor)
            matches.sort(key=lambda m:-m['similarity'])
            for candidate in matches:
                box=candidate['box'];x,y,X,Y=box;cx,cy=(x+X)/2,(y+Y)/2
                if any(box_overlap(box,q)>.25 for q in used):continue
                if any(l<=cx<=r and t-.15*(b-t)<=cy<=t+.55*(b-t) for l,t,r,b in others):continue
                support=[m for m in matches if m['seed']!=candidate['seed'] and box_overlap(m['box'],box)>.6]
                if not support or any(box_overlap(m['box'],box)<.2 for m in matches if m['similarity']>=candidate['similarity']-.025):continue
                confidence=verify(im,box,verifier)
                if confidence<.9:continue
                evidence[key].append(dict(frame=n,body=body,head=box,seeds=[candidate['seed'],support[0]['seed']],confidence=confidence,similarity=candidate['similarity']))
                l,t,r,b=body;w=r-l;h=b-t;rect=[max(0,int(l-.3*w)),max(0,int(t-.4*h)),min(im.width,int(r+.3*w)),min(im.height,int(b+.2*h))]
                tile=im.crop(rect);draw=ImageDraw.Draw(tile)
                for coords,col in ((body,"yellow"),(box,"lime")):
                    a,c,A,C=coords;draw.rectangle((a-rect[0],c-rect[1],A-rect[0],C-rect[1]),outline=col,width=4)
                tile.thumbnail((154,150));tiles[key].append((n,tile));break
    if tiles:
        out=Image.new("RGB",(640,len(tiles)*175))
        for j,k in enumerate(sorted(tiles)):
            ls=tiles[k]
            for i,idx in enumerate(sorted({round(v*(len(ls)-1)/3) for v in range(4)})):
                n,im=ls[idx];x=i*160;y=j*175;out.paste(im,(x,y));ImageDraw.Draw(out).text((x,y+151),f"{k} / {n}")
        out.save(OUT/"temporal-candidates.jpg",quality=94)
    (OUT/"temporal-candidates.json").write_text(json.dumps(evidence,indent=2))
    print({k:len(v) for k,v in evidence.items()})
if __name__=="__main__":main()
