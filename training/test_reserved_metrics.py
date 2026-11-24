import unittest
import torch
from evaluate_reserved import counts_at_threshold


class ReservedMetricsTests(unittest.TestCase):
    def test_duplicates_wrong_classes_and_threshold(self):
        gt={'boxes':torch.tensor([[0.,0.,10.,10.]]),'labels':torch.tensor([3])}
        pred={'boxes':gt['boxes'].repeat(4,1),'labels':torch.tensor([3,3,4,3]),'scores':torch.tensor([.9,.8,.7,.2])}
        rows=counts_at_threshold([pred],[gt])
        self.assertEqual((rows[2]['tp'],rows[2]['fp'],rows[2]['fn']),(1,1,0))
        self.assertEqual(rows[2]['precision'],.5)
        self.assertEqual(rows[3]['fp'],1)
        self.assertIsNone(rows[0]['precision'])


if __name__=='__main__':unittest.main()
