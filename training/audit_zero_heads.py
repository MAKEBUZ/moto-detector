"""Cache the unchanged per-frame detector output and sample the 24 original misses."""
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw
from predict import analyze_image, detector_objects
from traffic_video import Stream, Tracks
from photo_results import summarize
import torch

OUT=Path("reports/training/wide-zero-heads")
PREVIOUS=Path("reports/training/wide-temporal-check/video.json")

@torch.inference_mode()
def main():
    control=json.loads(PREVIOUS.read_text(encoding="utf8"))
    out=OUT;out.mkdir(exist_ok=True,parents=True)
    stream=Stream(control["source"],control["run"])
    ids=[t["id"] for t in control["trayectorias_ocupantes"] if not t["votos"]]
    tracker=Tracks();samples={k:[] for k in ids}
    with (out/"baseline.jsonl").open("w", encoding="utf8") as log:
        for n in range(1,control["frames"]+1):
            ok,f=stream.cap.read();assert ok
            im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
            base=analyze_image(stream.model,im,draw_boxes=False,video_heads=True)["predictions"]
            ps=stream.wide_heads.refine(im,base,lambda v,c:detector_objects(*stream.model[:3],v,c))
            assoc=summarize(ps);mapping=tracker.update(assoc["ocupantes"],n)
            for rider in assoc["ocupantes"]:
                k=mapping[rider["id"]]
                if k in samples:samples[k].append((n,rider["box"]))
            log.write(json.dumps(dict(frame=n,objects=ps),separators=(",",":"))+"\n")
    for t in control["trayectorias_ocupantes"]:
        tr=tracker.items[t["id"]];assert tr["hits"]==t["detecciones"] and dict(tr["votes"])==t["votos"]
    for index in (0,1,2):
        sheet=Image.new("RGB",(6*160,4*175))
        for i,k in enumerate(ids):
            obs=samples[k];n,box=obs[[0,len(obs)//2,-1][index]]
            stream.cap.set(1,n-1);ok,f=stream.cap.read();assert ok
            im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
            l,t,r,b=box;w=r-l;h=b-t
            ImageDraw.Draw(im).rectangle(box,outline="yellow",width=3)
            crop=im.crop((max(0,int(l-.4*w)),max(0,int(t-.4*h)),min(im.width,int(r+.4*w)),min(im.height,int(b+.1*h))))
            crop.thumbnail((158,150));x=(i%6)*160;y=(i//6)*175;sheet.paste(crop,(x,y));ImageDraw.Draw(sheet).text((x,y+150),f"{k} / {n}")
        sheet.save(out/f"tracks-{index}.jpg",quality=92)
    stream.cap.release()
    print(len(ids),sum(map(len,samples.values())),flush=True)

if __name__=="__main__":main()
