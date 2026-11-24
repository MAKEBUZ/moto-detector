import unittest
import torch
from models import build_detector


class ProposalBudgetTests(unittest.TestCase):
    def test_only_proposal_limits_change(self):
        torch.manual_seed(123);a=build_detector(5,8)
        torch.manual_seed(123);b=build_detector(5,8,4)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        for train,pre,post in [(True,4000,2000),(False,2000,800)]:
            b.train(train)
            self.assertEqual(b.rpn.pre_nms_top_n(),pre)
            self.assertEqual(b.rpn.post_nms_top_n(),post)
        self.assertEqual(a.rpn.nms_thresh,b.rpn.nms_thresh)
        self.assertEqual(b.roi_heads.detections_per_img,100)


if __name__=='__main__':unittest.main()
