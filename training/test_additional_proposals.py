import unittest
import torch
from audit_additional_proposals import matched_status


class MatchingTests(unittest.TestCase):
    def test_one_to_one_and_class_confusion(self):
        target={'boxes':torch.tensor([[0.,0,10,10],[20.,20,30,30]]),'labels':torch.tensor([4,3])}
        pred={'boxes':torch.tensor([[0.,0,10,10],[0.,0,10,10],[20.,20,30,30],[50.,50,60,60]]),
              'labels':torch.tensor([4,4,4,4]),'scores':torch.tensor([.9,.8,.7,.1])}
        self.assertEqual(matched_status(pred,target,.05),{0:'tp',1:'duplicate',2:'helmet_confusion',3:'other_region'})
        self.assertNotIn(3,matched_status(pred,target,.5))

    def test_no_nohelmet_ground_truth(self):
        target={'boxes':torch.tensor([[0.,0,10,10]]),'labels':torch.tensor([3])}
        pred={'boxes':torch.tensor([[0.,0,10,10]]),'labels':torch.tensor([4]),'scores':torch.tensor([.8])}
        self.assertEqual(matched_status(pred,target,.5),{0:'helmet_confusion'})


if __name__=='__main__':unittest.main()
