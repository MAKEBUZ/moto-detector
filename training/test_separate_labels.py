import unittest
from training.prepare_separate_labels import convert_scene


class SeparateLabelTests(unittest.TestCase):
    def test_no_implicit_training_or_incomplete_image(self):
        with self.assertRaises(ValueError): convert_scene({'split': 'val'})
        with self.assertRaises(ValueError): convert_scene({'split': 'train', 'annotation_complete': False})

    def test_motorcycle_uses_new_box_not_context(self):
        s = dict(image_id='sample', image='unused', sha256='unused', width=100, height=100, split='train', group_id='g', annotation_complete=True,
                 entities=[{'id': 'm', 'kind': 'motorcycle', 'bbox': [0, 50, 100, 100], 'context_bbox': [0, 0, 100, 100], 'review_status': 'confirmed'}], links=[])
        self.assertEqual(convert_scene(s)['annotations'][0]['box'], [0, 50, 100, 100])
        s['split'] = 'val'
        with self.assertRaises(ValueError): convert_scene(s)
        self.assertEqual(convert_scene(s, 'val')['split'], 'val')
        s['entities'][0]['review_status'] = 'uncertain'
        with self.assertRaises(ValueError): convert_scene(s, 'val')


if __name__ == '__main__': unittest.main()
