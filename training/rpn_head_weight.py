"""Extra RPN objectness loss on sampled positives matched to annotated heads."""
import torch
import torch.nn.functional as F


def install_head_weight(rpn):
    assign = rpn.assign_targets_to_anchors
    sampler = rpn.fg_bg_sampler
    compute = rpn.compute_loss
    context = {}
    stats = dict(updates=0, sampled_head_positives=0)

    def assign_with_heads(anchors, targets):
        labels, matched_boxes = assign(anchors, targets)
        masks = []
        for label, matched, target in zip(labels, matched_boxes, targets):
            positive = torch.where(label == 1)[0]
            heads = target["boxes"][target["labels"] >= 3]
            mask = torch.zeros_like(label, dtype=torch.bool)
            mask[positive] = (matched[positive,None] == heads[None]).all(-1).any(-1)
            masks.append(mask)
        context["heads"] = torch.cat(masks)
        return labels, matched_boxes

    def sample(labels):
        positive, negative = sampler(labels)
        context["selected"] = torch.cat(positive).bool()
        context["total"] = sum(p.sum() + n.sum() for p,n in zip(positive,negative))
        return positive, negative

    def compute_weighted(logits, deltas, labels, targets):
        objectness, regression = compute(logits, deltas, labels, targets)
        mask = context.pop("heads") & context.pop("selected")
        extra = F.softplus(-logits.flatten()[mask]).sum() / context.pop("total")
        stats["updates"] += 1
        stats["sampled_head_positives"] += int(mask.sum())
        return objectness + extra, regression

    rpn.assign_targets_to_anchors = assign_with_heads
    rpn.fg_bg_sampler = sample
    rpn.compute_loss = compute_weighted
    return stats
