import json
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageOps
from models import HelmetCNN,build_detector,forbid_external_weights
from data import HelmetDataset,collate_detection
from association import associate,iou


def load(task,device):
    ck=torch.load(Path('runs')/f'{task}-baseline'/'best.pt',map_location='cpu',weights_only=True)
    assert ck['config']['pretrained'] is False
    model=HelmetCNN() if task=='classifier' else build_detector()
    model.load_state_dict(ck['model']);return model.to(device).eval()


def match(preds,truth,threshold):
    used=set(); matches=[];false=[]
    for i,p in sorted(enumerate(preds),key=lambda x:-x[1]['score']):
        if p['score']<threshold:continue
        candidates=[(iou(p['box'],g['box']),j) for j,g in enumerate(truth) if j not in used and g['class_id']==p['class_id']]
        score,j=max(candidates,default=(0,None))
        if score>=.5:used.add(j);matches.append((i,j))
        else:false.append(i)
    return matches,false,[j for j in range(len(truth)) if j not in used]


@torch.inference_mode()
def main():
    torch.set_num_threads(4);forbid_external_weights();device='cuda' if torch.cuda.is_available() else 'cpu'
    out=Path('reports/error-review');out.mkdir(parents=True,exist_ok=True)
    dataset='data/prepared/helmet-baseline-v1'
    detector=load('detector',device);ds=HelmetDataset(dataset,'detector','val',augment=False)
    loader=DataLoader(ds,batch_size=2,num_workers=0,collate_fn=collate_detection)
    records=[]
    for images,targets in loader:
        predictions=detector([x.to(device) for x in images])
        for tensor,t,p in zip(images,targets,predictions):
            r=ds.rows[int(t['image_id'])];h,w=tensor.shape[-2:];sx=r['width']/w;sy=r['height']/h
            preds=[{'class_id':int(c)-1,'score':float(s),'box':[float(v)*scale for v,scale in zip(b,[sx,sy,sx,sy])]} for b,c,s in zip(p['boxes'].cpu(),p['labels'].cpu(),p['scores'].cpu())]
            truth=[{'class_id':a['classId'],'box':a['box']} for a in r['annotations']]
            matches,fp,fn=match(preds,truth,.5)
            records.append({'id':r['id'],'image':r['image'],'predictions':preds,'ground_truth':truth,'matches':matches,'false_positive_indices':fp,'false_negative_indices':fn,'association':associate(preds)})
    (out/'detector-validation.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    thresholds={}
    for threshold in (.3,.5,.7):
        totals=[Counter() for _ in range(3)]
        for r in records:
            pairs,fp,fn=match(r['predictions'],r['ground_truth'],threshold)
            for _,j in pairs:totals[r['ground_truth'][j]['class_id']]['tp']+=1
            for i in fp:totals[r['predictions'][i]['class_id']]['fp']+=1
            for j in fn:totals[r['ground_truth'][j]['class_id']]['fn']+=1
        thresholds[str(threshold)]=[{**dict(c),'precision':c['tp']/max(1,c['tp']+c['fp']),'recall':c['tp']/max(1,c['tp']+c['fn'])} for c in totals]
    size_counts={k:Counter() for k in ['under_24','24_to_47','48_plus']}
    for r in records:
        matched={j for _,j in r['matches']}
        for j,g in enumerate(r['ground_truth']):
            if g['class_id']==0:continue
            b=g['box'];size=min(b[2]-b[0],b[3]-b[1]);key='under_24' if size<24 else '24_to_47' if size<48 else '48_plus'
            size_counts[key]['total']+=1;size_counts[key]['detected']+=int(j in matched)
    statuses=Counter(a['status'] for r in records for a in r['association']['associations'])
    # Fixed first samples plus highest-error samples: explicit selection, not representative accuracy.
    chosen=sorted(records,key=lambda r:r['id'])[:4]
    chosen+= [r for r in sorted(records,key=lambda r:-(len(r['false_positive_indices'])+len(r['false_negative_indices']))) if r not in chosen][:8]
    queue=[]
    for r in chosen:
        image=Image.open(r['image']).convert('RGB');draw=ImageDraw.Draw(image)
        for i,p in enumerate(r['predictions']):
            if p['score']<.5:continue
            color='lime' if p['class_id']==0 else 'orange'
            draw.rectangle(p['box'],outline=color,width=3);draw.text(tuple(p['box'][:2]),f'{i}: {p["score"]:.2f}',fill=color)
        for a in r['association']['associations']:
            mi=a['motorcycle_index']
            if mi is None:continue
            head=r['predictions'][a['head_index']]['box'];moto=r['predictions'][mi]['box']
            draw.line(((head[0]+head[2])/2,(head[1]+head[3])/2,(moto[0]+moto[2])/2,(moto[1]+moto[3])/2),fill='yellow',width=3)
        image.thumbnail((1000,800));image.save(out/f'{r["id"]}.jpg')
        queue.append({'image_id':r['id'],'image':r['image'],'annotated_preview':f'{r["id"]}.jpg','proposals':r['association']['associations'],'reviewed':False,'human_links':None})
    (out/'association-review-queue.json').write_text(json.dumps(queue,indent=2))
    del detector
    model=load('classifier',device);cs=HelmetDataset(dataset,'classifier','val',augment=False)
    errors=[];offset=0
    for images,truth in DataLoader(cs,batch_size=64,num_workers=0):
        probs=model(images.to(device)).softmax(1).cpu();pred=probs.argmax(1)
        for k in range(len(truth)):
            if pred[k]!=truth[k]:errors.append({'id':cs.rows[offset+k]['id'],'original':cs.rows[offset+k]['image'],'true_class':int(truth[k]),'predicted_class':int(pred[k]),'confidence':float(probs[k].max())})
        offset+=len(truth)
    (out/'classifier-errors.json').write_text(json.dumps(sorted(errors,key=lambda r:-r['confidence']),indent=2))
    report={'split':'val','test_used':False,'detector_images':len(records),'classifier_examples':len(cs),'classifier_errors':len(errors),'threshold_metrics_iou50':thresholds,'head_recall_by_original_box_size':{k:{**dict(v),'recall':v['detected']/max(1,v['total'])} for k,v in size_counts.items()},'association_status_counts':dict(statuses),'association_accuracy':None,'association_ground_truth_available':False,'association_policy':'Geometric proposals only; never confirmed occupancy or counts.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
