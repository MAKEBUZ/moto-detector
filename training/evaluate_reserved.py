"""One frozen evaluation of the reviewed reserved batch; no training or tuning."""
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.ops import box_iou
from models import build_detector, forbid_external_weights
from metrics import detection_ap50
from overfit_heads import probe


def counts_at_threshold(predictions, targets, threshold=.5, overlap=.5):
    result=[]
    for cls in (1,2,3,4):
        tp=fp=fn=0
        for p,t in zip(predictions,targets):
            gt=t['boxes'][t['labels']==cls];used=set()
            ids=torch.where((p['labels']==cls)&(p['scores']>=threshold))[0]
            for i in sorted(ids.tolist(),key=lambda i:-float(p['scores'][i])):
                overlaps=box_iou(p['boxes'][i:i+1],gt)[0]
                choices=[j for j in range(len(gt)) if j not in used and overlaps[j]>=overlap]
                if choices:
                    used.add(max(choices,key=lambda j:float(overlaps[j])));tp+=1
                else:fp+=1
            fn+=len(gt)-len(used)
        result.append({'class_id':cls-1,'tp':tp,'fp':fp,'fn':fn,
                       'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None})
    return result


@torch.inference_mode()
def main():
    out=Path('reports/independent-evaluation/results-01')
    if out.exists(): raise FileExistsError('Reserved evaluation already exists; do not overwrite')
    protocol_path=Path('reports/independent-evaluation/protocol.json')
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    checkpoint=Path(protocol['checkpoint'])
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(checkpoint)==protocol['checkpoint_sha256']
    draft=Path('data/annotations/test-separate-v2-01/draft.json')
    assert sha(draft)==protocol['batch_sha256']
    root=Path('data/prepared/helmet-separate-v2-test-01')
    preparation=json.loads((root/'preparation.json').read_text())
    reviewed=Path('data/annotations/test-separate-v2-01/test-separate-v2-01-reviewed.json')
    assert preparation['reviewed_sha256']==sha(reviewed)
    rows=list(map(json.loads,(root/'test-manifest.jsonl').read_text().splitlines()))
    assert rows and set(r['image_id'] for r in rows)<=set(protocol['images'])
    decisions={r['image']:r for r in map(json.loads,Path('reports/preparation/image-decisions.jsonl').read_text().splitlines())}
    samples=[]
    for i,r in enumerate(rows):
        assert r['split']=='test' and decisions[r['image']]['split']=='test'
        assert sha(Path(r['image']))==r['sha256']
        with Image.open(r['image']) as im:
            image=im.convert('RGB');w,h=image.size
            assert (w,h)==(r['width'],r['height'])
            scale=min(1.,640/max(w,h));nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
            image=image.resize((nw,nh),Image.Resampling.BILINEAR)
        boxes=torch.tensor([a['box'] for a in r['annotations']],dtype=torch.float32)*torch.tensor([nw/w,nh/h,nw/w,nh/h])
        target={'boxes':boxes,'labels':torch.tensor([a['classId']+1 for a in r['annotations']]),'image_id':torch.tensor(i)}
        samples.append((pil_to_tensor(image).float()/255,target))
    torch.set_num_threads(4);forbid_external_weights()
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True);config=ck['config']
    assert config['pretrained'] is False and config['initialization']=='random'
    assert ck['epoch']==protocol['selected_epoch'] and config['classes']==protocol['classes']
    assert config['feature_stride']==protocol['feature_stride'] and config['proposal_multiplier']==protocol['proposal_multiplier']
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=build_detector(5,protocol['feature_stride'],protocol['proposal_multiplier']);model.load_state_dict(ck['model']);model.to(device).eval()
    predictions=[];details=[]
    for r,(image,target) in zip(rows,samples):
        p={k:v.cpu() for k,v in model([image.to(device)])[0].items()};predictions.append(p)
        details.append({'image_id':r['image_id'],'coordinates':'resized input pixels','predictions':{k:v.tolist() for k,v in p.items()},'target':{k:v.tolist() for k,v in target.items()}})
    metrics=detection_ap50(predictions,[t for _,t in samples],(1,2,3,4))
    operating=counts_at_threshold(predictions,[t for _,t in samples],protocol['confidence_threshold'],protocol['matching_iou'])
    diagnostic=probe(model,samples,device)
    result={'split':'test','images':len(rows),'excluded':preparation['excluded'],'classes':protocol['classes'],
            'metrics':metrics,'operating_point':operating,'confidence_threshold':protocol['confidence_threshold'],'matching_iou':protocol['matching_iou'],
            'head_rpn_coverage':{'covered':diagnostic['rpn_covered_iou50'],'total':diagnostic['head_count']},
            'protocol_sha256':sha(protocol_path),'checkpoint_sha256':sha(checkpoint),'review_sha256':sha(reviewed),
            'source_sha256':{str(p):sha(p) for p in (Path(__file__),Path('training/models.py'),Path('training/metrics.py'))},
            'training_performed':False,'thresholds_tuned':False,'limitation':'Small selected reserved subset, not representative traffic; now evaluated, not untouched test.'}
    out.mkdir(parents=True)
    for name,value in [('results.json',result),('predictions.json',details),('head-diagnostics.json',diagnostic)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
