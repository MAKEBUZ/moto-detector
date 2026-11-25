"""Random context crops centered on annotated no-helmet heads in auxiliary photos."""
import random
import torch
from data import collate_detection


def context_batch(samples):
    result=[]
    for image,target in samples:
        boxes=target["boxes"]
        heads=boxes[target["labels"]==2]
        if not len(heads) or random.random()>=.35:
            result.append((image,target))
            continue
        areas=(heads[:,2:]-heads[:,:2]).prod(1)
        weights=(1+(areas<areas.median()).int()).tolist()
        head=random.choices(heads.tolist(),weights=weights,k=1)[0]
        h,w=image.shape[-2:]
        fraction=random.uniform(.65,.9)
        cw=min(w,max(round(w*fraction),int(head[2]-head[0])+2))
        ch=min(h,max(round(h*fraction),int(head[3]-head[1])+2))
        x=max(0,min(w-cw,int((head[0]+head[2])/2-cw*random.uniform(.35,.65))))
        y=max(0,min(h-ch,int((head[1]+head[3])/2-ch*random.uniform(.35,.65))))
        shifted=boxes-boxes.new_tensor([x,y,x,y])
        shifted[:,0::2].clamp_(0,cw)
        shifted[:,1::2].clamp_(0,ch)
        keep=(shifted[:,2:]>shifted[:,:2]).all(1)
        result.append((image[:,y:y+ch,x:x+cw],dict(target,boxes=shifted[keep],labels=target["labels"][keep])))
    return collate_detection(result)
