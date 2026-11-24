import unittest
import torch
from models import build_detector


class FeatureStrideTests(unittest.TestCase):
    def test_resolution_and_initial_weights(self):
        torch.manual_seed(123);a=build_detector(5,16).eval()
        torch.manual_seed(123);b=build_detector(5,8).eval()
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        with torch.inference_mode():
            x=torch.zeros(1,3,128,128)
            self.assertEqual(a.backbone(x).shape[-2:],(8,8))
            self.assertEqual(b.backbone(x).shape[-2:],(16,16))
        self.assertEqual(a.rpn.anchor_generator.sizes,b.rpn.anchor_generator.sizes)
        with self.assertRaises(ValueError):build_detector(5,4)


if __name__=='__main__':unittest.main()
