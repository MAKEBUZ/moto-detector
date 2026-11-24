import unittest
from training.evaluate_relations import match_boxes, evaluate_scene


class RelationEvaluationTests(unittest.TestCase):
    def test_matching_is_one_to_one_and_class_agnostic(self):
        predictions = [{'class_id': c, 'score': score, 'box': [1, 1, 9, 9]} for c, score in [(1, .7), (2, .9)]]
        matches = match_boxes(predictions, [{'id': 'h', 'bbox': [1, 1, 9, 9]}], {1, 2})
        self.assertEqual(matches['h']['prediction_index'], 1)
        self.assertEqual(len(matches), 1)

    def test_context_semantics_and_conflict_abstention(self):
        scene = {'image_id': 'test', 'entities': [
            {'id': 'h', 'kind': 'head', 'bbox': [35, 5, 55, 25], 'review_status': 'confirmed', 'helmet': 'con_casco'},
            {'id': 'm', 'kind': 'motorcycle', 'bbox': [0, 60, 100, 100], 'review_status': 'confirmed'}],
            'links': [{'head_id': 'h', 'person_id': 'p', 'motorcycle_id': 'm', 'status': 'confirmed'}]}
        reference = {'entities': [{'id': 'm', 'context_bbox': [0, 0, 100, 100]}]}
        predictions = [{'class_id': 0, 'score': .9, 'box': [0, 0, 100, 100]},
                       {'class_id': 1, 'score': .9, 'box': [35, 5, 55, 25]}]
        result = evaluate_scene(scene, reference, predictions)
        self.assertEqual(result['links'][0]['outcome'], 'correct_candidate')
        predictions.append({'class_id': 2, 'score': .8, 'box': [35, 5, 55, 25]})
        result = evaluate_scene(scene, reference, predictions)
        self.assertEqual(result['links'][0]['outcome'], 'conflicting_head')
        self.assertFalse(result['counting_allowed'])
        scene['links'][0]['status'] = 'uncertain'
        self.assertEqual(evaluate_scene(scene, reference, predictions)['links'], [])


if __name__ == '__main__':
    unittest.main()
