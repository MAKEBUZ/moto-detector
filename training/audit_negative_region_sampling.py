"""Frozen proposal audit and controlled sampler replay on six reviewed train regions."""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
import torch
from torchvision.ops import box_iou
from data import SeparateHelmetDataset
from models import build_detector, forbid_external_weights
from train import fingerprint
from audit_positive_sampling import proposals


def summarize_draws(values):
    return {'mean':sum(values)/len(values),'zero_draws':values.count(0),'draws':values}


@torch.inference_mode()
def main():
    out=Path('reports/training/hard-negatives-fixed1950-01/negative-region-audit');out.mkdir(exist_ok=False)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    comparisons=read(out.parent/'comparison.json')['records']
    root=Path('data/prepared/helmet-separate-v2-expanded-04');ds=SeparateHelmetDataset(root,'train',augment=False)
    for name,h in read(root/'dataset.lock.json').items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
    selected=[i for i,r in enumerate(ds.rows) if r.get('reviewed_nonhead_regions')];assert len(selected)==6
    protocol={'models':[r['run'] for r in comparisons],'region_image_ids':[ds.rows[i]['image_id'] for i in selected],
              'split':'train','training_performed':False,'repeats':20,'region_iou_thresholds':[.3,.5],
              'eval':'eval mode, preRPN2000/post800; original selected train order batch2, no augmentations',
              'replay':'train mode including batch BN, preRPN4000/post2000, restore BN before each batch and at end; no gradients/optimizer. Same batches and RNG seeds for all models.',
              'caveat':'Final-weight controlled replay, NOT actual historical samples. Region means not-head, not necessarily RPN-background: motorcycle/person are valid positives. Controls44 never trained on these six scenes.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.set_num_threads(4);forbid_external_weights();device='cuda' if torch.cuda.is_available() else 'cpu';reports=[]
    for job in comparisons:
        path=Path('runs')/job['run']/'last.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==job['checkpoint_sha256']
        ck=torch.load(path,map_location='cpu',weights_only=True);c=ck['config'];assert not c['pretrained'] and c['initialization']=='random'
        model=build_detector(5,c['feature_stride'],c['proposal_multiplier']).to(device);model.load_state_dict(ck['model']);initial=fingerprint(model)
        buffers={n:b.clone() for n,b in model.named_buffers()}
        def restore():
            for n,b in model.named_buffers():b.copy_(buffers[n])
        records=[]
        for mode in ('eval','replay'):
            for start in range(0,6,2):
                restore();model.train(mode=='replay')
                with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device=='cuda' else []):
                    torch.manual_seed(73100+start)
                    indices=selected[start:start+2];samples=[ds[i] for i in indices];xs=[x.to(device) for x,_ in samples]
                    ts=[{k:v.to(device) for k,v in t.items()} for _,t in samples]
                    ims,targets=model.transform(xs,ts);features=OrderedDict([('0',model.backbone(ims.tensors))])
                    anchors,boxes,stages,obj=proposals(model,ims,features)
                    rpn_labels,_=model.rpn.assign_targets_to_anchors(anchors,targets)
                    augmented=model.roi_heads.add_gt_proposals(boxes,[t['boxes'] for t in targets])
                    matches,labels=model.roi_heads.assign_targets_to_proposals(augmented,[t['boxes'] for t in targets],[t['labels'] for t in targets])
                    if mode=='eval':
                        roi=model.roi_heads;logits,deltas=roi.box_predictor(roi.box_head(roi.box_roi_pool(features,boxes,ims.image_sizes)))
                        probs=logits.softmax(-1).split([len(b) for b in boxes]);final=roi.postprocess_detections(logits,deltas,boxes,ims.image_sizes)
                    else:
                        rp=[];rn=[];ri=[]
                        for _ in range(20):
                            p,n=model.rpn.fg_bg_sampler(rpn_labels);rp.append(p);rn.append(n);ri.append(model.roi_heads.subsample(labels))
                        # Check direct assignment/subsampling against torchvision's complete selector.
                        state=torch.random.get_rng_state();cuda=torch.cuda.get_rng_state() if device=='cuda' else None
                        expected=model.roi_heads.subsample(labels);torch.random.set_rng_state(state)
                        if cuda is not None:torch.cuda.set_rng_state(cuda)
                        actual,_,actual_labels,_=model.roi_heads.select_training_samples(boxes,targets)
                        for i in range(len(xs)):
                            assert torch.equal(actual[i],augmented[i][expected[i]]) and torch.equal(actual_labels[i],labels[i][expected[i]])
                    for i,index in enumerate(indices):
                        row=ds.rows[index];region=row['reviewed_nonhead_regions'][0]
                        h,w=ims.image_sizes[i];box=torch.tensor([region['bbox']],device=device)*torch.tensor([w/row['width'],h/row['height'],w/row['width'],h/row['height']],device=device)
                        record={'image_id':row['image_id'],'mode':mode,'category':region['category'],'region_box_model':box[0].tolist(),
                                'gt_iou': [{'label':int(l),'iou':float(v)} for l,v in zip(targets[i]['labels'],box_iou(box,targets[i]['boxes'])[0])],
                                'stage_max_iou':{name:float(box_iou(box,b).max()) if len(b) else 0. for name,b in stages[i].items()},'thresholds':{}}
                        for threshold in (.3,.5):
                            near_anchor=box_iou(box,anchors[i])[0]>=threshold
                            near_roi=box_iou(box,augmented[i])[0]>=threshold
                            data={'rpn_available':{str(k):int(((rpn_labels[i]==k)&near_anchor).sum()) for k in (-1,0,1)},
                                  'roi_available':{str(k):int(((labels[i]==k)&near_roi).sum()) for k in (-1,0,1,2,3,4)},
                                  'roi_injected_near':int(near_roi[len(boxes[i]):].sum())}
                            if mode=='replay':
                                data['rpn_selected']={str(k):summarize_draws([int((mask[i].bool()&near_anchor).sum()) for mask in masks]) for k,masks in ((0,rn),(1,rp))}
                                data['roi_selected']={str(k):summarize_draws([int((near_roi[ids[i]]&(labels[i][ids[i]]==k)).sum()) for ids in ri]) for k in (0,1,2,3,4)}
                            else:
                                near=near_roi[:len(boxes[i])];data['proposal_count']=int(near.sum())
                                data['argmax_counts']=torch.bincount(probs[i][near].argmax(1),minlength=5).tolist()
                                data['max_probabilities']=probs[i][near].max(0).values.tolist() if near.any() else None
                                final_near=box_iou(box,final[0][i])[0]>=threshold
                                data['final_heads']=[{'label':int(final[2][i][j]),'score':float(final[1][i][j]),'box':final[0][i][j].tolist()} for j in torch.where(final_near&(final[2][i]>=3))[0].tolist()]
                            record['thresholds'][str(threshold)]=data
                        records.append(record)
        restore();model.eval();assert fingerprint(model)==initial and hashlib.sha256(path.read_bytes()).hexdigest()==digest
        report={'run':job['run'],'images':job['images'],'seed':job['seed'],'checkpoint_sha256':digest,'state_unchanged':True,'records':records}
        (out/(job['run']+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8');reports.append(report);print(job['run']+' done',flush=True)
        del model,ck
    (out/'comparison.json').write_text(json.dumps({'reports':reports},indent=2),encoding='utf-8')


if __name__=='__main__':main()
