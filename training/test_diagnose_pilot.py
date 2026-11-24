import unittest
from training.diagnose_pilot import inspect_box


class DiagnosticTests(unittest.TestCase):
    def test_low_confidence_differs_from_bad_localization(self):
        box = [0, 0, 10, 10]
        low = {'class_id': 1, 'score': .2, 'box': box}
        far = {'class_id': 2, 'score': .99, 'box': [20, 20, 30, 30]}
        self.assertEqual(inspect_box(box, [low, far], {1, 2})['status'], 'below_confidence_threshold')
        self.assertEqual(inspect_box(box, [far], {1, 2})['status'], 'no_matching_box_in_cache')
        self.assertEqual(inspect_box(box, [{**low, 'score': .5}], {1, 2})['status'], 'localized')
        self.assertEqual(inspect_box(box, [low], {0})['status'], 'no_matching_box_in_cache')


if __name__ == '__main__':
    unittest.main()
