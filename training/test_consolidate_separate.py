import unittest
from training.consolidate_separate_dataset import assert_disjoint


class ConsolidationTests(unittest.TestCase):
    def test_split_leakage_and_duplicates(self):
        a = dict(image_id='a', sha256='a', group_id='g1', split='train')
        b = dict(image_id='b', sha256='b', group_id='g2', split='val')
        assert_disjoint([a, b])
        for key, value in [('image_id', 'a'), ('sha256', 'a'), ('group_id', 'g1')]:
            with self.assertRaises(ValueError): assert_disjoint([a, {**b, key: value}])


if __name__ == '__main__': unittest.main()
