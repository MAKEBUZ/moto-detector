"""Frozen-checkpoint proposal stages and controlled positive-sampler replay."""
import hashlib
import argparse
import json
from collections import OrderedDict
from pathlib import Path
import torch
from torchvision.ops import box_iou, clip_boxes_to_image, remove_small_boxes, batched_nms
from torchvision.models.detection.rpn import concat_box_prediction_layers
from data import SeparateHelmetDataset
from models import build_detector, forbid_external_weights
from metrics import detection_ap50
from train import fingerprint


def max_iou(gt, boxes):
    return box_iou(gt, boxes).max(1).values if len(boxes) else torch.zeros(len(gt), device=gt.device)


def proposals(model, ims, features):
    rpn = model.rpn
    maps = list(features.values())
    obj, delta = rpn.head(maps)
    counts = [o[0].numel() for o in obj]
    anchors = rpn.anchor_generator(ims, maps)
    obj, delta = concat_box_prediction_layers(obj, delta)
    decoded = rpn.box_coder.decode(delta, anchors).reshape(len(anchors), -1, 4)
    logits = obj.reshape(len(anchors), -1)
    top = rpn._get_top_n_idx(logits, counts)
    final, _ = rpn.filter_proposals(decoded, obj, ims.image_sizes, counts)
    stages = []
    for i, shape in enumerate(ims.image_sizes):
        # This project's own backbone exposes one feature level.
        assert len(counts) == 1
        all_boxes = clip_boxes_to_image(decoded[i], shape)
        top_boxes = all_boxes[top[i]]
        boxes = top_boxes; scores = logits[i, top[i]].sigmoid()
        keep = remove_small_boxes(boxes, rpn.min_size)
        boxes, scores = boxes[keep], scores[keep]
        keep = torch.where(scores >= rpn.score_thresh)[0]
        boxes, scores = boxes[keep], scores[keep]
        keep = batched_nms(boxes, scores, torch.zeros_like(scores, dtype=torch.long), rpn.nms_thresh)
        nms = boxes[keep]
        assert torch.equal(nms[:rpn.post_nms_top_n()], final[i])
        stages.append({'anchors': anchors[i], 'decoded_clipped': all_boxes,
                       'topk_clipped': top_boxes, 'topk_filtered': boxes, 'after_nms_before_limit': nms, 'final': final[i]})
    return anchors, final, stages, logits


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/training/fixed-updates-01/positive-sampling'))
    out = parser.parse_args().output
    out.mkdir(exist_ok=False)
    protocol = json.loads(Path('reports/training/fixed-updates-01/protocol.json').read_text())
    torch.set_num_threads(4); forbid_external_weights()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reports = []
    for job in protocol['jobs']:
        path = Path('runs')/job['run']/'last.pt'
        checkpoint_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        ck = torch.load(path, map_location='cpu', weights_only=True)
        c = ck['config']; assert not c['pretrained'] and c['initialization']=='random'
        assert ck['optimizer_updates']==1950
        root = Path(job['dataset'])
        for n,digest in json.loads((root/'dataset.lock.json').read_text()).items():
            assert hashlib.sha256((root/n).read_bytes()).hexdigest()==digest
        model = build_detector(5, c['feature_stride'], c['proposal_multiplier']).to(device)
        model.load_state_dict(ck['model']); initial = fingerprint(model)
        # Includes BN statistics; restore before every diagnostic batch.
        saved_buffers = {n:b.clone() for n,b in model.named_buffers()}
        def restore_buffers():
            for n,b in model.named_buffers(): b.copy_(saved_buffers[n])
        val_records = []; train_records = []; predictions = []; truths = []
        for split in ('val','train'):
            ds = SeparateHelmetDataset(root, split, augment=False)
            for start in range(0,len(ds),2):
                restore_buffers()
                model.train(split=='train')
                torch.manual_seed(91000+start)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(91000+start)
                samples = [ds[i] for i in range(start,min(start+2,len(ds)))]
                xs = [x.to(device) for x,_ in samples]
                ts = [{k:v.to(device) for k,v in t.items()} for _,t in samples]
                if split=='val':
                    pred = model(xs)
                    predictions.extend([{k:v.cpu() for k,v in p.items()} for p in pred])
                    truths.extend([t for _,t in samples])
                ims, targets = model.transform(xs, ts)
                features = OrderedDict([('0', model.backbone(ims.tensors))])
                anchors, boxes, stages, objectness_logits = proposals(model, ims, features)
                if split=='val':
                    pooled=model.roi_heads.box_roi_pool(features,boxes,ims.image_sizes)
                    logits,_=model.roi_heads.box_predictor(model.roi_heads.box_head(pooled))
                    probs=logits.softmax(-1).split([len(b) for b in boxes])
                    for i,t in enumerate(targets):
                        row=ds.rows[start+i]
                        coverage={name:max_iou(t['boxes'],b) for name,b in stages[i].items()}
                        for j,label in enumerate(t['labels'].tolist()):
                            if label not in (3,4):continue
                            overlaps=box_iou(t['boxes'][j:j+1],boxes[i])[0]
                            covered=overlaps>=.5
                            q=int(overlaps.argmax()) if len(overlaps) else None
                            original=samples[i][1]['boxes'][j:j+1].to(device)
                            correct=pred[i]['labels']==label
                            strong=correct & (pred[i]['scores']>=.5)
                            all_overlaps=box_iou(t['boxes'][j:j+1],stages[i]['decoded_clipped'])[0]
                            all_cover=all_overlaps>=.5
                            best_cover_score=objectness_logits[i][all_cover].max() if all_cover.any() else None
                            rank=int((objectness_logits[i]>best_cover_score).sum())+1 if best_cover_score is not None else None
                            val_records.append({'image_id':row['image_id'],'entity_id':row['annotations'][j]['entity_id'],'label':label,
                                'stage_max_iou':{k:float(v[j]) for k,v in coverage.items()},
                                'best_covering_proposal_objectness_rank':rank,
                                'best_covering_proposal_objectness':float(best_cover_score.sigmoid()) if best_cover_score is not None else None,
                                'pre_nms_budget':model.rpn.pre_nms_top_n(),
                                'final_proposals_covering_gt':int(covered.sum()),
                                'covering_proposal_class_counts':torch.bincount(probs[i][covered].argmax(1),minlength=5).tolist(),
                                'best_overlap_proposal_probs':probs[i][q].tolist() if q is not None else None,
                                'correct_final_iou':float(max_iou(original,pred[i]['boxes'][correct])[0]),
                                'correct_final_iou_score50':float(max_iou(original,pred[i]['boxes'][strong])[0])})
                else:
                    # Reproduce assignment + samplers directly, without optimizer or losses.
                    rpn_labels,_=model.rpn.assign_targets_to_anchors(anchors,targets)
                    rpn_matches=[model.rpn.proposal_matcher(box_iou(t['boxes'],a)) for t,a in zip(targets,anchors)]
                    gt_boxes=[t['boxes'] for t in targets];gt_labels=[t['labels'] for t in targets]
                    augmented=model.roi_heads.add_gt_proposals(boxes,gt_boxes)
                    matches,labels=model.roi_heads.assign_targets_to_proposals(augmented,gt_boxes,gt_labels)
                    repeats=20
                    samples_rpn=[];samples_roi=[]
                    for repeat in range(repeats):
                        rp,_=model.rpn.fg_bg_sampler(rpn_labels)
                        ri=model.roi_heads.subsample(labels)
                        samples_rpn.append(rp);samples_roi.append(ri)
                    for i,t in enumerate(targets):
                        row=ds.rows[start+i];n=len(boxes[i])
                        # Validate directly against select_training_samples using a shared RNG snapshot.
                        state=torch.random.get_rng_state();cuda_state=torch.cuda.get_rng_state() if device=='cuda' else None
                        expected=model.roi_heads.subsample(labels)
                        torch.random.set_rng_state(state)
                        if cuda_state is not None:torch.cuda.set_rng_state(cuda_state)
                        actual,actual_matches,actual_labels,_=model.roi_heads.select_training_samples(boxes,targets)
                        assert torch.equal(actual[i],augmented[i][expected[i]])
                        assert torch.equal(actual_labels[i],labels[i][expected[i]])
                        for j,label in enumerate(t['labels'].tolist()):
                            positive=(labels[i]>0)&(matches[i]==j)
                            rp_positive=(rpn_labels[i]==1)&(rpn_matches[i]==j)
                            rpn_counts=[int((m[i].bool()&rp_positive).sum()) for m in samples_rpn]
                            roi_counts=[int(positive[ids[i]].sum()) for ids in samples_roi]
                            injected_counts=[int(((ids[i]>=n)&positive[ids[i]]).sum()) for ids in samples_roi]
                            train_records.append({'image_id':row['image_id'],'entity_id':row['annotations'][j]['entity_id'],'label':label,
                                'rpn_available_positive_anchors':int(rp_positive.sum()),
                                'rpn_sampled_mean':sum(rpn_counts)/repeats,'rpn_zero_draws':rpn_counts.count(0),
                                'roi_positive_proposals_before_gt':int(positive[:n].sum()),
                                'roi_positive_after_gt':int(positive.sum()),
                                'roi_sampled_mean':sum(roi_counts)/repeats,'roi_zero_draws':roi_counts.count(0),
                                'roi_sampled_injected_mean':sum(injected_counts)/repeats,'draws':repeats})
        restore_buffers()
        assert fingerprint(model)==initial,'Diagnostic changed model state'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==checkpoint_sha
        metric=detection_ap50(predictions,truths,(1,2,3,4))
        assert abs(metric['map50']-ck['validation']['map50'])<1e-6
        aggregate={}
        for cls in (1,2,3,4):
            rows=[r for r in train_records if r['label']==cls]
            aggregate[str(cls)]={'gt_count':len(rows),
                'rpn_available_positive_anchors':sum(r['rpn_available_positive_anchors'] for r in rows),
                'rpn_sampled_mean_total':sum(r['rpn_sampled_mean'] for r in rows),
                'roi_positive_before_gt':sum(r['roi_positive_proposals_before_gt'] for r in rows),
                'roi_sampled_mean_total':sum(r['roi_sampled_mean'] for r in rows),
                'roi_injected_sampled_mean_total':sum(r['roi_sampled_injected_mean'] for r in rows),
                'gt_without_proposal_before_injection':sum(r['roi_positive_proposals_before_gt']==0 for r in rows),
                'rpn_zero_head_draws':sum(r['rpn_zero_draws'] for r in rows),
                'roi_zero_head_draws':sum(r['roi_zero_draws'] for r in rows),'draws_per_gt':20}
        report={**job,'checkpoint_sha256':checkpoint_sha,'model_state_unchanged':True,
                'validation_metrics_reproduced':metric,'validation_heads':val_records,
                'training_sampler_replay':train_records,'training_by_class':aggregate,
                'sampler_settings':{'rpn_batch_size':model.rpn.fg_bg_sampler.batch_size_per_image,
                                    'rpn_positive_fraction':model.rpn.fg_bg_sampler.positive_fraction,
                                    'roi_batch_size':model.roi_heads.fg_bg_sampler.batch_size_per_image,
                                    'roi_positive_fraction':model.roi_heads.fg_bg_sampler.positive_fraction}}
        (out/(job['run']+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
        reports.append(report)
        print(job['run']+' done',flush=True)
        del model,ck
    (out/'summary.json').write_text(json.dumps({'reports':reports,
        'method':'Validation eval mode batch 2; training replay train mode including batch BN, fixed manifest order, no augmentations, 20 sampler draws per image. BN buffers restored before every batch and at end. No gradients, optimizer or weight updates.',
        'limitations':'This is a controlled replay at final weights, not the historical samples of 1950 training steps. Train order/augmentations differ from training. RPN is binary object/background; class attribution follows matched GT. RoI adds GT boxes before sampling; injected positives do not prove proposal generalization.',
        'test_used':False},indent=2),encoding='utf-8')


if __name__=='__main__':main()
