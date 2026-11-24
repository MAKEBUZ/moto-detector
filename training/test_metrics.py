import unittest
import torch
from metrics import classifier_metrics, detection_ap50


class MetricTests(unittest.TestCase):
    def test_classifier_class_order_and_zero_recall(self):
        m = classifier_metrics(torch.tensor([[5, 0], [5, 0]]))
        self.assertEqual(m['recall'], [1., 0.])
        self.assertAlmostEqual(m['accuracy'], .5)
        self.assertAlmostEqual(m['macro_f1'], 1/3, places=6)

    def test_perfect_detection(self):
        target = {'boxes': torch.tensor([[0., 0., 10., 10.]]), 'labels': torch.tensor([1])}
        prediction = {**target, 'scores': torch.tensor([.9])}
        self.assertAlmostEqual(detection_ap50([prediction], [target])['map50'], 1.)

    def test_wrong_class_is_not_a_match(self):
        target = {'boxes': torch.tensor([[0., 0., 10., 10.]]), 'labels': torch.tensor([1])}
        prediction = {'boxes': target['boxes'], 'labels': torch.tensor([2]), 'scores': torch.tensor([.9])}
        self.assertEqual(detection_ap50([prediction], [target])['map50'], 0.)

    def test_one_ground_truth_cannot_match_twice(self):
        target = {'boxes': torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]]), 'labels': torch.tensor([1, 1])}
        prediction = {'boxes': torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]]), 'labels': torch.tensor([1, 1]), 'scores': torch.tensor([.9, .8])}
        self.assertAlmostEqual(detection_ap50([prediction], [target])['map50'], 51/101, places=6)


if __name__ == '__main__':
    unittest.main()
