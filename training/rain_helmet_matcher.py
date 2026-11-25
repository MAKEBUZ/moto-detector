"""Appearance check for dark helmets in the rain footage using labeled local examples."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from photo_results import summarize, box_overlap


class RainHelmetMatcher:
    def __init__(self):
        root=Path(__file__).resolve().parents[1]/"data/prepared/rain-rider-02"
        self.templates=[]
        for line in (root/"train-manifest.jsonl").read_text().splitlines():
            row=json.loads(line)
            head=next(a["box"] for a in row["annotations"] if a["classId"]==2)
            l,t,r,b=head
            with Image.open(root/row["image"]) as im:
                patch=im.convert("L").crop((l-.18*(r-l),t-.15*(b-t),r+.18*(r-l),b+.15*(b-t))).resize((24,28),Image.Resampling.BILINEAR)
            self.templates.append(np.asarray(patch))

    def recover(self,im,objects):
        result=list(objects)
        for rider in summarize(objects)["ocupantes"]:
            if rider["estado"]!="indeterminado":continue
            l,t,r,b=rider["box"];h=b-t;w=r-l
            # Bound work per rider independently of the source video resolution.
            x,y=int(l-.12*w),int(t-.18*h)
            crop=im.crop((x,y,int(r+.12*w),int(t+.46*h))).convert("L")
            if min(crop.size)<8:continue
            factor=112/crop.height
            view=np.asarray(crop.resize((max(1,round(crop.width*factor)),112),Image.Resampling.BILINEAR))
            best=None
            for H in (23,32,42,54):
                W=round(24*H/28)
                if W>view.shape[1]:continue
                maps=[cv2.matchTemplate(view,cv2.resize(p,(W,H)),cv2.TM_CCOEFF_NORMED) for p in self.templates]
                # Require two labeled views to support the same location.
                scores=np.partition(np.stack(maps),-2,axis=0)[-2]
                _,val,_,loc=cv2.minMaxLoc(scores)
                if best is None or val>best[0]:
                    X,Y=loc
                    best=(val,[x+(X+.13*W)/factor,y+(Y+.115*H)/factor,x+(X+.87*W)/factor,y+(Y+.885*H)/factor])
            if best is None or best[0]<.74:continue
            score,box=best
            if any(p["class_id"]>=2 and box_overlap(p["box"],box)>.25 for p in result):continue
            proposed=result+[dict(class_id=2,**{"class":"cabeza_con_casco"},score=score,box=box,source="rain_helmet_appearance")]
            if summarize(proposed)["ocupantes"][rider["id"]-1]["estado"]=="con_casco":result=proposed
        return result
