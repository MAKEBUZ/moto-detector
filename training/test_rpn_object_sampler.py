import unittest
import torch
from torchvision.models.detection._utils import BalancedPositiveNegativeSampler
from training.rpn_object_sampler import reserve_objects, ObjectPositiveSampler, install_object_sampler
from training.models import build_detector


class ReservationTests(unittest.TestCase):
    def test_reserves_missing_objects_without_changing_budget(self):
        labels=torch.tensor([1,1,1,1,1,0,-1.])
        matched=torch.tensor([0,0,0,1,2,-1,-2])
        mask=torch.tensor([1,1,1,0,0,0,0],dtype=torch.uint8)
        out,available,before,after=reserve_objects(mask,labels,matched,4,torch.Generator().manual_seed(4))
        self.assertEqual(out.sum(),3)
        self.assertEqual(after.tolist(),[1,1,1,0])
        self.assertEqual(available.tolist(),[3,1,1,0])
        self.assertEqual(before.tolist(),[3,0,0,0])
        self.assertFalse(bool(out[-2:].any()))

    def test_budget_shortfall_is_explicit(self):
        with self.assertRaises(ValueError):
            reserve_objects(torch.tensor([1,0],dtype=torch.uint8),torch.ones(2),torch.tensor([0,1]),2,torch.Generator())

    def test_no_objects(self):
        out,*_=reserve_objects(torch.zeros(3,dtype=torch.uint8),torch.zeros(3),torch.full((3,),-1),0,torch.Generator())
        self.assertEqual(out.sum(),0)

    def test_negative_selection_and_global_rng_preserved(self):
        base=BalancedPositiveNegativeSampler(8,.5)
        labels=[torch.tensor([1.]*20+[1.]+[0.]*30)]
        matched=torch.tensor([0]*20+[1]+[-1]*30)
        custom=ObjectPositiveSampler(base,999)
        custom.context=([matched],[{'boxes':torch.zeros(2,4),'labels':torch.tensor([1,4]),'image_id':torch.tensor(0)}])
        torch.manual_seed(3);p,n=base(labels);expected_rng=torch.random.get_rng_state()
        torch.manual_seed(3);pp,nn=custom(labels)
        self.assertTrue(torch.equal(expected_rng,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(n[0],nn[0]))
        self.assertEqual(p[0].sum(),pp[0].sum())
        self.assertTrue(all(o['selected']>=1 for o in custom.records[0]['objects']))

    def test_install_keeps_weights_and_eval_outputs(self):
        torch.set_num_threads(2);torch.manual_seed(2)
        model=build_detector(5,8,4).eval()
        before={k:v.clone() for k,v in model.state_dict().items()}
        x=[torch.rand(3,64,64)]
        with torch.no_grad(): expected=model(x)
        install_object_sampler(model,111)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        with torch.no_grad(): actual=model(x)
        self.assertTrue(all(torch.equal(expected[0][k],actual[0][k]) for k in expected[0]))


if __name__=='__main__':unittest.main()
