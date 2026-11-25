"""Passive observation of actual sampler outputs; no additional random draws."""
from types import MethodType
import torch
from torchvision.ops import box_iou


class ObservedSampler:
    def __init__(self, original, owner, stage):
        self.original=original;self.owner=owner;self.stage=stage;self.context=None
        self.batch_size_per_image=original.batch_size_per_image
        self.positive_fraction=original.positive_fraction

    def __call__(self, labels):
        positive,negative=self.original(labels)
        boxes,targets=self.context
        with torch.no_grad():
            for i,(b,t,lab,p,n) in enumerate(zip(boxes,targets,labels,positive,negative)):
                regions=t.get('negative_regions')
                if regions is None or not len(regions):continue
                overlap=box_iou(regions,b.detach())
                before=n.clone()
                if self.owner.reserve:
                    # Preserve positives and quotas; replace only background selections.
                    protected=torch.zeros_like(n,dtype=torch.bool)
                    for row in overlap:
                        candidates=torch.where((row>=.5)&(lab==0))[0]
                        if not len(candidates):continue
                        chosen=candidates[torch.argmax(row[candidates])]
                        if n[chosen]:protected[chosen]=True;continue
                        removable=torch.where(n.bool()&~protected)[0]
                        if not len(removable):continue
                        # Remove the selected background least related to any reviewed region.
                        relevance=overlap[:,removable].max(dim=0).values
                        victim=removable[torch.argmin(relevance)]
                        n[victim]=0;n[chosen]=1;protected[chosen]=True
                    assert int(n.sum())==int(before.sum())
                    assert not bool((n.bool()&(lab!=0)).any())
                for j,region in enumerate(regions):
                    details={}
                    for threshold in (.3,.5):
                        near=overlap[j]>=threshold
                        selected=p.bool()|n.bool()
                        details[str(threshold)]={
                            'available':{str(k):int((near&(lab==k)).sum()) for k in (-1,0,1,2,3,4)},
                            'selected':{str(k):int((near&selected&(lab==k)).sum()) for k in (-1,0,1,2,3,4)},
                            'selected_indices':torch.where(near&selected)[0].cpu().tolist()}
                    self.owner.records.append({'stage':self.stage,'image_index':int(t['image_id']),'region_index':j,
                        'region_box_model':region.cpu().tolist(),'total_candidates':len(b),'reserved_negative_changes':int((n!=before).sum())//2,
                        'total_selected_positive':int(p.sum()),'total_selected_negative':int(n.sum()),'thresholds':details})
        self.context=None
        return positive,negative


class ExposureObserver:
    def __init__(self, model, reserve=False):
        self.records=[];self.reserve=reserve
        # torchvision transforms standard boxes only; resize diagnostic metadata separately.
        resize=model.transform.resize
        def observed_resize(transform,image,target=None):
            old_shape=image.shape[-2:]
            image,target=resize(image,target)
            if target is not None and 'negative_regions' in target:
                h,w=image.shape[-2:];oh,ow=old_shape
                target['negative_regions']=target['negative_regions']*target['negative_regions'].new_tensor([w/ow,h/oh,w/ow,h/oh])
            return image,target
        model.transform.resize=MethodType(observed_resize,model.transform)
        rpn=model.rpn;sampler=ObservedSampler(rpn.fg_bg_sampler,self,'rpn');rpn.fg_bg_sampler=sampler
        assign=rpn.assign_targets_to_anchors
        def rpn_assign(module,anchors,targets):
            result=assign(anchors,targets);sampler.context=(anchors,targets);return result
        rpn.assign_targets_to_anchors=MethodType(rpn_assign,rpn)
        roi=model.roi_heads;roi_sampler=ObservedSampler(roi.fg_bg_sampler,self,'roi');roi.fg_bg_sampler=roi_sampler
        select=roi.select_training_samples;roi_assign=roi.assign_targets_to_proposals
        current={}
        def select_samples(module,proposals,targets):
            current['targets']=targets
            try:return select(proposals,targets)
            finally:current.clear()
        def assign_proposals(module,proposals,gt_boxes,gt_labels):
            result=roi_assign(proposals,gt_boxes,gt_labels)
            roi_sampler.context=(proposals,current['targets']);return result
        roi.select_training_samples=MethodType(select_samples,roi)
        roi.assign_targets_to_proposals=MethodType(assign_proposals,roi)
