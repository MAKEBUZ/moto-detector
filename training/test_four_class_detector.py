import unittest
import torch
from data import SeparateHelmetDataset
from models import build_detector
from metrics import detection_ap50


class FourClassTests(unittest.TestCase):
    def test_fourth_class_is_measured(self):
        target = {'boxes': torch.tensor([[0., 0., 10., 10.]]), 'labels': torch.tensor([4])}
        pred = {**target, 'scores': torch.tensor([.9])}
        result = detection_ap50([pred], [target], (1, 2, 3, 4))
        self.assertEqual(len(result['per_class']), 4)
        self.assertAlmostEqual(result['per_class'][3]['ap50'], 1.)
        self.assertAlmostEqual(result['map50'], 1.)

    def test_loader_and_output_contract(self):
        dataset = SeparateHelmetDataset('data/prepared/helmet-separate-v2', 'val')
        self.assertFalse(dataset.augment)
        labels = set()
        for i in range(len(dataset)):
            image, target = dataset[i]
            labels.update(target['labels'].tolist())
            self.assertTrue((target['boxes'][:, 2:] > target['boxes'][:, :2]).all())
            self.assertTrue((target['boxes'][:, [0, 2]] <= image.shape[2]+.001).all())
            self.assertTrue((target['boxes'][:, [1, 3]] <= image.shape[1]+.001).all())
        self.assertEqual(labels, {1, 2, 3, 4})
        self.assertEqual(build_detector(5).roi_heads.box_predictor.cls_score.out_features, 5)


if __name__ == '__main__': unittest.main()
