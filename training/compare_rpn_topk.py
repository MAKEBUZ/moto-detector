"""Frozen 2000/4000 pre-NMS RPN budget comparison with paired timing."""
import hashlib
import json
import statistics
import time
from pathlib import Path
import torch
from torchvision.ops import box_iou
from data import SeparateHelmetDataset
from models import build_detector, forbid_external_weights
from metrics import detection_ap50
from evaluate_reserved import counts_at_threshold
from train import fingerprint


@torch.inference_mode()
def main():
    out=Path('reports/training/object-reserve-01/topk-2000-vs-4000')
    out.mkdir(exist_ok=False)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    pairs=read(Path('reports/training/object-reserve-01/comparison.json'))['records']
    torch.set_num_threads(4);forbid_external_weights();device='cuda' if torch.cuda.is_available() else 'cpu'
    jobs=[{'run':run,'variant':variant,'images':p['images'],'seed':p['seed']} for p in pairs for variant,run in [('control',p['control_run']),('reserve',p['run'])]]
    protocol={'jobs':jobs,'pre_nms_budgets':[2000,4000],'post_nms_budget':800,'checkpoint':'last.pt',
              'confidence_thresholds':[.05,.5],'matching_iou':.5,'warmup_passes_per_budget':3,'paired_repetitions':20,
              'timing':'Wall-clock perf_counter, CUDA synchronize before/after complete five-image pass. GPU inputs preloaded; excludes file loading, CPU->GPU transfer, metrics, hooks and report generation. Includes model transform/backbone/RPN/RoI/postprocess. Batches 2,2,1; alternating order each repetition and initial order each model.',
              'selection':'Report all eight checkpoints, no budget/seed selection or production changes.',
              'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
              'torch':torch.__version__,'cpu_threads':4,'training_performed':False,'test_used':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    results=[]
    for job_index,job in enumerate(jobs):
        path=Path('runs')/job['run']/'last.pt';checkpoint_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        ck=torch.load(path,map_location='cpu',weights_only=True);c=ck['config'];root=Path(c['dataset'])
        assert not c['pretrained'] and c['initialization']=='random' and ck['optimizer_updates']==1950
        for name,h in read(root/'dataset.lock.json').items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
        ds=SeparateHelmetDataset(root,'val',augment=False);samples=[ds[i] for i in range(len(ds))]
        targets=[t for _,t in samples];gpu_inputs=[x.to(device) for x,_ in samples]
        batches=[gpu_inputs[i:i+2] for i in range(0,len(ds),2)]
        model=build_detector(5,c['feature_stride'],c['proposal_multiplier']).to(device)
        model.load_state_dict(ck['model']);model.eval();initial=fingerprint(model)
        assert model.rpn._pre_nms_top_n['testing']==2000 and model.rpn._post_nms_top_n['testing']==800
        pre_original=dict(model.rpn._pre_nms_top_n)
        def settings():return {'pre_nms_train':model.rpn._pre_nms_top_n['training'],
            'post_nms':dict(model.rpn._post_nms_top_n),'rpn_nms':model.rpn.nms_thresh,
            'rpn_score':model.rpn.score_thresh,'roi_nms':model.roi_heads.nms_thresh,
            'roi_score':model.roi_heads.score_thresh,'final_limit':model.roi_heads.detections_per_img}
        fixed=settings();variants={};pred_details={}
        for budget in (2000,4000):
            model.rpn._pre_nms_top_n['testing']=budget
            assert settings()==fixed
            captured=[]
            hook=model.rpn.register_forward_hook(lambda module,inputs,output:captured.extend([p.detach().clone() for p in output[0]]))
            preds=[]
            try:
                for batch in batches:preds.extend([{k:v.cpu() for k,v in p.items()} for p in model(batch)])
            finally:hook.remove()
            assert len(captured)==len(ds)
            heads=[]
            for start,batch in zip(range(0,len(ds),2),batches):
                _,truth=model.transform(batch,[{k:v.to(device) for k,v in t.items()} for t in targets[start:start+2]])
                for i,t in enumerate(truth):
                    boxes=captured[start+i]
                    for j,cls in enumerate(t['labels'].tolist()):
                        if cls not in (3,4):continue
                        overlap=float(box_iou(t['boxes'][j:j+1],boxes).max()) if len(boxes) else 0.
                        heads.append({'image_id':ds.rows[start+i]['image_id'],'entity_id':ds.rows[start+i]['annotations'][j]['entity_id'],
                                      'label':cls,'post_nms_max_iou':overlap})
            metrics=detection_ap50(preds,targets,(1,2,3,4))
            if budget==2000:assert abs(metrics['map50']-ck['validation']['map50'])<1e-6
            variants[str(budget)]={'metrics':metrics,'counts':{str(th):counts_at_threshold(preds,targets,th) for th in (.05,.5)},
                 'actual_post_nms_counts':[len(p) for p in captured],'heads':heads,
                 'nohelmet_post_nms_coverage':sum(h['label']==4 and h['post_nms_max_iou']>=.5 for h in heads)}
            pred_details[str(budget)]=[{'image_id':r['image_id'],'prediction':{k:v.tolist() for k,v in p.items()},
                                       'target':{k:v.tolist() for k,v in t.items()}} for r,p,t in zip(ds.rows,preds,targets)]
        def synchronize():
            if device=='cuda':torch.cuda.synchronize()
        def timed_pass(budget):
            model.rpn._pre_nms_top_n['testing']=budget
            synchronize();start=time.perf_counter()
            for batch in batches:outputs=model(batch)
            synchronize();elapsed=(time.perf_counter()-start)*1000
            del outputs
            return elapsed
        warm_order=(2000,4000) if job_index%2==0 else (4000,2000)
        for budget in warm_order:
            for _ in range(3):timed_pass(budget)
        timings={2000:[],4000:[]};paired=[]
        for repeat in range(20):
            order=(2000,4000) if (repeat+job_index)%2==0 else (4000,2000)
            elapsed={b:timed_pass(b) for b in order}
            for b,v in elapsed.items():timings[b].append(v)
            paired.append({'repeat':repeat,'order':list(order),'ms_2000':elapsed[2000],'ms_4000':elapsed[4000]})
        for budget,values in timings.items():
            sorted_values=sorted(values)
            variants[str(budget)]['timing']={'full_pass_ms':values,'median_pass_ms':statistics.median(values),
                'median_amortized_ms_per_image':statistics.median(values)/len(ds),
                'p10_pass_ms':float(torch.quantile(torch.tensor(sorted_values),.1)),
                'p90_pass_ms':float(torch.quantile(torch.tensor(sorted_values),.9))}
        model.rpn._pre_nms_top_n=pre_original
        assert settings()==fixed and fingerprint(model)==initial and hashlib.sha256(path.read_bytes()).hexdigest()==checkpoint_hash
        result={**job,'checkpoint_sha256':checkpoint_hash,'fixed_settings_verified':fixed,'weights_unchanged':True,
                'variants':variants,'paired_timings':paired,
                'median_paired_runtime_change_pct':statistics.median(100*(p['ms_4000']/p['ms_2000']-1) for p in paired)}
        results.append(result)
        (out/(job['run']+'.json')).write_text(json.dumps({'summary':result,'predictions':pred_details},indent=2),encoding='utf-8')
        print(json.dumps({'run':job['run'],'AP_nohelmet':{b:v['metrics']['per_class'][3]['ap50'] for b,v in variants.items()},
                         'runtime_change_pct':result['median_paired_runtime_change_pct']}),flush=True)
        del ck,model
    (out/'comparison.json').write_text(json.dumps({'results':results,'protocol_sha256':hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),
        'timing_warning':'Amortized batch inference on five preloaded images, not single-frame latency, end-to-end video FPS or a production benchmark. GPU load/thermal state and allocator noise may affect results.',
        'training_performed':False,'test_used':False,'deployment_changed':False},indent=2),encoding='utf-8')


if __name__=='__main__':main()
