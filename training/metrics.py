import torch
from torchvision.ops import box_iou


def classifier_metrics(confusion):
    m = confusion.float()
    tp = m.diag()
    precision = tp / m.sum(0).clamp(min=1)
    recall = tp / m.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
    return {'accuracy': float(tp.sum() / m.sum().clamp(min=1)), 'macro_f1': float(f1.mean()),
            'precision': precision.tolist(), 'recall': recall.tolist(), 'f1': f1.tolist(),
            'confusion_matrix_true_rows': confusion.tolist()}


def detection_ap50(predictions, targets, class_ids=(1, 2, 3)):
    """101-point AP at IoU .50, all areas, up to 100 predictions/image. NOT COCO mAP50:95."""
    aps = []
    details = []
    for cls in class_ids:
        gt = [t['boxes'][t['labels'] == cls] for t in targets]
        count = sum(len(x) for x in gt)
        entries = []
        for i, pred in enumerate(predictions):
            for box, score in zip(pred['boxes'][pred['labels'] == cls], pred['scores'][pred['labels'] == cls]):
                entries.append((float(score), i, box))
        entries.sort(key=lambda x: -x[0])
        used = [set() for _ in gt]
        tp, fp = [], []
        for score, i, box in entries:
            match = False
            if len(gt[i]):
                overlaps = box_iou(box[None], gt[i])[0]
                for k in sorted(range(len(overlaps)), key=lambda k: -float(overlaps[k])):
                    if float(overlaps[k]) < .5:
                        break
                    if k not in used[i]:
                        used[i].add(k)
                        match = True
                        break
            tp.append(int(match)); fp.append(int(not match))
        if tp and count:
            tp_sum = torch.tensor(tp).cumsum(0)
            fp_sum = torch.tensor(fp).cumsum(0)
            recall = tp_sum / count
            precision = tp_sum / (tp_sum + fp_sum)
            ap = sum(float(precision[recall >= t].max()) if (recall >= t).any() else 0.
                     for t in torch.linspace(0, 1, 101)) / 101
        else:
            ap = 0.
        if count:
            aps.append(ap)
        details.append({'class_id': cls - 1, 'ap50': ap, 'ground_truth_boxes': count})
    return {'map50': sum(aps) / max(1, len(aps)), 'per_class': details,
            'definition': '101-point AP at IoU=0.50; not COCO mAP50:95'}
