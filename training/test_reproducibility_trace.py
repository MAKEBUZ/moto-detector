import unittest
import torch
from torchvision.models.detection._utils import BalancedPositiveNegativeSampler
from reproducibility_trace import SamplerTrace, tensors_hash, rng_hashes


class TraceTests(unittest.TestCase):
    def test_passive_sampler_and_hashes(self):
        sampler=BalancedPositiveNegativeSampler(8,.5);records=[];trace=SamplerTrace(sampler,records,'rpn')
        labels=[torch.tensor([1]*8+[0]*16+[-1]*3)]
        torch.manual_seed(8);state=torch.get_rng_state();expected=sampler(labels);end=torch.get_rng_state()
        torch.set_rng_state(state);actual=trace(labels)
        self.assertTrue(torch.equal(end,torch.get_rng_state()))
        for a,b in zip(expected,actual):self.assertTrue(torch.equal(a[0],b[0]))
        self.assertEqual(records[0]['positive_sha256'],tensors_hash(expected[0]))
        before=rng_hashes();tensors_hash(labels);self.assertEqual(before,rng_hashes())


if __name__=='__main__':unittest.main()
