"""Reserve a positive RPN anchor per matched object without changing budgets."""
import torch
from torchvision.models.detection.rpn import RegionProposalNetwork


def reserve_objects(mask, labels, matches, num_objects, generator):
    result = mask.clone()
    available = torch.bincount(matches[labels == 1], minlength=num_objects)
    before = torch.bincount(matches[mask.bool()], minlength=num_objects)
    if int((available > 0).sum()) > int(mask.sum()):
        raise ValueError('Positive budget cannot cover every eligible object')
    current = before.clone()
    for obj in torch.where((available > 0) & (current == 0))[0].tolist():
        donors = torch.where(result.bool() & (current[matches.clamp(min=0)] > 1))[0]
        candidates = torch.where((labels == 1) & (matches == obj))[0]
        remove = donors[torch.randint(len(donors), (1,), device=mask.device, generator=generator)[0]]
        add = candidates[torch.randint(len(candidates), (1,), device=mask.device, generator=generator)[0]]
        result[remove] = 0; result[add] = 1
        current[matches[remove]] -= 1; current[obj] += 1
    assert int(result.sum()) == int(mask.sum())
    assert bool((labels[result.bool()] == 1).all())
    assert bool((current[available > 0] >= 1).all())
    return result, available, before, current


class ObjectPositiveSampler:
    def __init__(self, original, seed):
        self.original = original
        self.batch_size_per_image = original.batch_size_per_image
        self.positive_fraction = original.positive_fraction
        self.seed = seed
        self.generators = {}
        self.context = None
        self.records = []

    def __call__(self, labels):
        positive, negative = self.original(labels)
        assert self.context is not None
        matches, targets = self.context
        for i, (mask, lab, matched, target) in enumerate(zip(positive, labels, matches, targets)):
            device = str(mask.device)
            if device not in self.generators:
                self.generators[device] = torch.Generator(device=mask.device).manual_seed(self.seed)
            new, available, before, after = reserve_objects(mask, lab, matched, len(target['boxes']), self.generators[device])
            positive[i] = new
            ids = torch.where(new.bool())[0]
            matched_selected = matched[ids]
            self.records.append({'image_index': int(target['image_id']),
                'positive_count': int(new.sum()), 'negative_count': int(negative[i].sum()),
                'objects': [{'gt_index': j, 'class_id': int(target['labels'][j])-1,
                             'available': int(available[j]), 'baseline_selected': int(before[j]),
                             'selected': int(after[j]),
                             'anchor_indices': ids[matched_selected == j].cpu().tolist()}
                            for j in range(len(target['boxes']))]})
        self.context = None
        return positive, negative


class ObjectReservedRPN(RegionProposalNetwork):
    def assign_targets_to_anchors(self, anchors, targets):
        result = super().assign_targets_to_anchors(anchors, targets)
        if self.training:
            matches = [self.proposal_matcher(self.box_similarity(t['boxes'], a))
                       if len(t['boxes']) else torch.full((len(a),), -1, device=a.device, dtype=torch.long)
                       for a,t in zip(anchors,targets)]
            self.fg_bg_sampler.context = (matches, targets)
        return result


def install_object_sampler(model, seed):
    assert type(model.rpn) is RegionProposalNetwork
    model.rpn.__class__ = ObjectReservedRPN
    model.rpn.fg_bg_sampler = ObjectPositiveSampler(model.rpn.fg_bg_sampler, seed)
    return model.rpn.fg_bg_sampler
