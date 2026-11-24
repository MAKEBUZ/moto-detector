import copy
import json
import unittest
from training.validate_annotation_batch import DEFAULT, validate


class AnnotationValidationTests(unittest.TestCase):
    def setUp(self):
        self.ref = json.loads(DEFAULT.read_text(encoding='utf-8'))
        self.batch = copy.deepcopy(self.ref)

    def result(self, reviewed=False):
        return validate(self.batch, self.ref, reviewed=reviewed, verify_images=False)

    def reviewed_fixture(self):
        self.batch.update(human_reviewed=True, reviewer='SYNTHETIC TEST ONLY', reviewed_at='2026-11-23T12:00:00+00:00')
        for s in self.batch['scenes']:
            s['reviewed'] = True
            for e in s['entities']:
                e['review_status'] = 'uncertain'
            for l in s['links']:
                l['status'] = 'uncertain'

    def test_draft_is_structurally_valid_but_not_reviewed(self):
        self.assertTrue(self.result()['valid'])
        self.assertFalse(self.result(True)['valid'])

    def test_split_and_source_cannot_change(self):
        for field, value in [('split', 'train'), ('sha256', 'changed'), ('group_id', 'changed')]:
            self.batch = copy.deepcopy(self.ref)
            self.batch['scenes'][0][field] = value
            self.assertFalse(self.result()['valid'])

    def test_bad_boxes(self):
        for box in [[0, 0, float('nan'), 5], [0, 0, True, 5], [3, 3, 2, 2], [-1, 0, 4, 5]]:
            self.batch['scenes'][0]['entities'][0]['bbox'] = box
            self.assertFalse(self.result()['valid'])

    def test_dangling_reference(self):
        self.batch['scenes'][0]['links'][0]['person_id'] = 'absent'
        self.assertFalse(self.result()['valid'])

    def test_uncertain_links_are_excluded(self):
        self.reviewed_fixture()
        result = self.result(True)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['confirmed_links'], [])
        self.assertFalse(result['training_ready'])

    def test_confirmed_link_requires_all_boxes(self):
        self.reviewed_fixture()
        s = self.batch['scenes'][0]
        for e in s['entities']:
            e.update(review_status='confirmed', bbox=e['bbox'] or [1, 1, 10, 10])
        s['links'][0]['status'] = 'confirmed'
        self.assertEqual(len(self.result(True)['confirmed_links']), 1)
        s['entities'][0]['bbox'] = None
        self.assertFalse(self.result(True)['valid'])

    def test_background_person_cannot_belong_to_motorcycle(self):
        self.reviewed_fixture()
        s = next(s for s in self.batch['scenes'] if s['image_id'] == 'Image_01720')
        for e in s['entities']:
            e.update(review_status='confirmed', bbox=e['bbox'] or [1, 1, 10, 10])
        s['links'][-1].update(status='confirmed', motorcycle_id='m0')
        self.assertFalse(self.result(True)['valid'])


if __name__ == '__main__':
    unittest.main()
