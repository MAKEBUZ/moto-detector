import copy
import unittest
import torch
import random
from pathlib import Path
from data import SeparateHelmetDataset
from models import build_detector
from negative_exposure import ExposureObserver


class ExposureTests(unittest.TestCase):
    def test_dataset_metadata_follows_flip_without_changing_augmentation(self):
        root=Path('data/prepared/helmet-separate-v2-expanded-04')
        if not root.exists():self.skipTest('Reviewed dataset unavailable')
        plain=SeparateHelmetDataset(root,'train',photometric=True,photo_seed=91)
        logged=SeparateHelmetDataset(root,'train',photometric=True,photo_seed=91);logged.exposure_regions=True
        index=next(i for i,r in enumerate(plain.rows) if r.get('reviewed_nonhead_regions'))
        state=random.getstate()
        try:
            random.seed(1);a,t=plain[index];after=random.getstate()
            random.seed(1);b,u=logged[index]
            self.assertEqual(after,random.getstate());self.assertTrue(torch.equal(a,b))
            for k in t:self.assertTrue(torch.equal(t[k],u[k]),k)
            row=plain.rows[index];h,w=a.shape[-2:]
            box=torch.tensor([row['reviewed_nonhead_regions'][0]['bbox']])*torch.tensor([w/row['width'],h/row['height'],w/row['width'],h/row['height']])
            box[:,[0,2]]=w-box[:,[2,0]]
            self.assertTrue(torch.equal(box,u['negative_regions']))
        finally:random.setstate(state)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_forward_rng_and_actual_selections(self):
        torch.manual_seed(71);torch.cuda.manual_seed_all(71);torch.set_num_threads(4)
        baseline=build_detector(5,8,4).cuda().train();observed=copy.deepcopy(baseline);observer=ExposureObserver(observed)
        captured={}
        class Capture:
            def __init__(self,original,name):self.original=original;self.name=name
            def __call__(self,labels):
                result=self.original(labels);captured[self.name]=[[x.clone() for x in group] for group in result];return result
        for stage in ('rpn','roi_heads'):
            left=getattr(baseline,stage);right=getattr(observed,stage)
            left.fg_bg_sampler=Capture(left.fg_bg_sampler,'base'+stage)
            right.fg_bg_sampler.original=Capture(right.fg_bg_sampler.original,'logged'+stage)
        xs=[torch.rand(3,100,120,device='cuda') for _ in range(2)]
        ts=[{'boxes':torch.tensor([[10.,10,90,90],[35.,15,50,30]],device='cuda'),'labels':torch.tensor([2,4],device='cuda'),'image_id':torch.tensor(i,device='cuda')} for i in range(2)]
        cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state();a=baseline(xs,ts);end=torch.get_rng_state();endgpu=torch.cuda.get_rng_state()
        torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
        b=observed(xs,[{**t,'negative_regions':torch.tensor([[35.,60,50,75]],device='cuda')} for t in ts])
        self.assertTrue(torch.equal(end,torch.get_rng_state()));self.assertTrue(torch.equal(endgpu,torch.cuda.get_rng_state()))
        for k in a:self.assertTrue(torch.equal(a[k],b[k]),k)
        for stage in ('rpn','roi_heads'):
            for group1,group2 in zip(captured['base'+stage],captured['logged'+stage]):
                for mask1,mask2 in zip(group1,group2):self.assertTrue(torch.equal(mask1,mask2))

    def test_observer_preserves_losses_gradients_and_rng(self):
        torch.set_num_threads(2);torch.manual_seed(31)
        baseline=build_detector(5,8,4).train();observed=copy.deepcopy(baseline);observer=ExposureObserver(observed)
        images=[torch.rand(3,100,120),torch.rand(3,100,120)]
        targets=[{'boxes':torch.tensor([[10.,10,90,90],[35.,15,50,30]]),'labels':torch.tensor([2,4]),'image_id':torch.tensor(i)} for i in range(2)]
        extra=[{**t,'negative_regions':torch.tensor([[35.,60,50,75]])} for t in targets]
        state=torch.random.get_rng_state();a=baseline(images,targets);sum(a.values()).backward();end=torch.random.get_rng_state()
        torch.random.set_rng_state(state);b=observed(images,extra);sum(b.values()).backward()
        self.assertTrue(torch.equal(end,torch.random.get_rng_state()))
        for k in a:self.assertTrue(torch.equal(a[k],b[k]),k)
        for (name,p),(other,q) in zip(baseline.named_parameters(),observed.named_parameters()):
            self.assertEqual(name,other);self.assertTrue(torch.equal(p.grad,q.grad),name)
        for k,v in baseline.state_dict().items():self.assertTrue(torch.equal(v,observed.state_dict()[k]),k)
        self.assertEqual(len(observer.records),4)
        transformed,_=baseline.transform(images,targets)
        h,w=transformed.image_sizes[0]
        expected=torch.tensor([35.,60,50,75])*torch.tensor([w/120,h/100,w/120,h/100])
        for r in observer.records:
            self.assertTrue(torch.equal(torch.tensor(r['region_box_model']),expected))
            for d in r['thresholds'].values():self.assertEqual(sum(d['selected'].values()),len(d['selected_indices']))


if __name__=='__main__':unittest.main()
