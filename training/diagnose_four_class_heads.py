"""Frozen-checkpoint diagnostics: final heads, RPN coverage, oracle-box class scores."""
import hashlib
import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path
import torch
from torchvision.ops import box_iou
from data import SeparateHelmetDataset
from models import build_detector, forbid_external_weights


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/prepared/helmet-separate-v2'))
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/detector-separate-v2-pilot-01/best.pt'))
    parser.add_argument('--output', type=Path, default=Path('reports/training/separate-v2-pilot-01/head-diagnostics'))
    args = parser.parse_args()
    torch.set_num_threads(4); forbid_external_weights()
    root = args.dataset
    checkpoint = args.checkpoint
    if not checkpoint.resolve().is_relative_to(Path('runs').resolve()):
        raise ValueError('Only project checkpoints allowed')
    for name, digest in json.loads((root/'dataset.lock.json').read_text()).items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest: raise ValueError('Dataset changed')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert ck['config']['pretrained'] is False and ck['config']['initialization'] == 'random'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_detector(5, ck['config'].get('feature_stride',16), ck['config'].get('proposal_multiplier',1)); model.load_state_dict(ck['model']); model.to(device).eval()
    records = []; summary = {}
    for split in ('train', 'val'):
        ds = SeparateHelmetDataset(root, split, augment=False)
        counts = Counter(); sizes = []
        for i in range(len(ds)):
            image, target = ds[i]; image = image.to(device)
            prediction = model([image])[0]
            images, targets = model.transform([image], [{k: v.to(device) for k,v in target.items()}])
            features = model.backbone(images.tensors)
            if isinstance(features, torch.Tensor): features = OrderedDict([('0', features)])
            proposals, _ = model.rpn(images, features, targets)
            gt = targets[0]
            pooled = model.roi_heads.box_roi_pool(features, [gt['boxes']], images.image_sizes)
            logits, _ = model.roi_heads.box_predictor(model.roi_heads.box_head(pooled))
            probs = logits.softmax(-1)
            for j, label in enumerate(target['labels'].tolist()):
                if label not in (3,4): continue
                counts['heads'] += 1
                box = target['boxes'][j].to(device)
                overlaps = box_iou(box[None], prediction['boxes'])[0]
                def best(mask):
                    inds = torch.where(mask)[0]
                    if not len(inds): return None
                    k = int(inds[overlaps[inds].argmax()])
                    return {'iou': float(overlaps[k]), 'score': float(prediction['scores'][k]), 'label': int(prediction['labels'][k])}
                head_best = best((prediction['labels']==3)|(prediction['labels']==4))
                correct_best = best(prediction['labels']==label)
                any_best = best(prediction['labels']>0)
                rpn_iou = float(box_iou(gt['boxes'][j:j+1], proposals[0]).max()) if len(proposals[0]) else 0.
                head_ok = head_best is not None and head_best['iou'] >= .5
                correct_ok = correct_best is not None and correct_best['iou'] >= .5
                counts['head_box_coverage_iou50'] += int(head_ok)
                counts['correct_class_box_coverage_iou50'] += int(correct_ok)
                counts['rpn_coverage_iou50'] += int(rpn_iou >= .5)
                counts['oracle_box_class_correct'] += int(probs[j].argmax()==label)
                side = float((gt['boxes'][j,2:]-gt['boxes'][j,:2]).min()); sizes.append(side)
                records.append({'split': split, 'image_id': ds.rows[i]['image_id'], 'entity_id': ds.rows[i]['annotations'][j]['entity_id'],
                                'true_label': label, 'min_side_model_pixels': side, 'best_head_prediction': head_best,
                                'best_correct_class_prediction': correct_best, 'best_any_class_prediction': any_best,
                                'best_rpn_iou': rpn_iou, 'oracle_box_class_probabilities': probs[j].cpu().tolist()})
        summary[split] = {**dict(counts), 'min_head_side_model_pixels': min(sizes), 'median_head_side_model_pixels': sorted(sizes)[len(sizes)//2],
                          'heads_under_16_pixels': sum(x<16 for x in sizes), 'heads_under_32_pixels': sum(x<32 for x in sizes)}
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    report = {'summary': summary, 'records': records, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'epoch': ck['epoch'], 'model_score_threshold': model.roi_heads.score_thresh, 'model_nms_threshold': model.roi_heads.nms_thresh,
              'limits': ['Coverage is per reference box, not one-to-one detection AP.', 'Oracle box classification uses known target boxes, not end-to-end detection.',
                         'RPN coverage measured after proposal filtering; low coverage does not isolate anchor versus learned proposal failures.',
                         'No parameter updates; no test data; no threshold tuning.'], 'test_used': False}
    (out/'diagnosis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
