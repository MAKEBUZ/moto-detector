import unittest
from association import associate

def p(cls,box,score=.9):return {'class_id':cls,'box':box,'score':score}
class AssociationTests(unittest.TestCase):
    def test_candidate_is_never_confirmed_occupancy(self):
        r=associate([p(0,[0,0,100,200]),p(1,[40,10,60,40])])
        self.assertEqual(r['associations'][0]['status'],'candidate_requires_review')
        self.assertFalse(r['occupancy_confirmed']);self.assertFalse(r['counting_allowed'])
    def test_outside_head_is_unassigned(self):
        r=associate([p(0,[0,0,100,200]),p(2,[140,10,160,40])])
        self.assertEqual(r['associations'][0]['status'],'unassigned')
    def test_overlapping_motorcycles_abstain(self):
        r=associate([p(0,[0,0,100,200]),p(0,[5,0,105,200]),p(1,[40,10,60,40])])
        self.assertEqual(r['associations'][0]['status'],'ambiguous')
    def test_passengers_are_not_limited_to_one_head(self):
        r=associate([p(0,[0,0,100,200]),p(1,[20,10,40,40]),p(2,[60,30,80,60])])
        self.assertEqual([a['motorcycle_index'] for a in r['associations']],[0,0])
    def test_conflicting_head_classes_abstain(self):
        r=associate([p(0,[0,0,100,200]),p(1,[40,10,60,40]),p(2,[40,10,60,40],.8)])
        self.assertTrue(all(a['status']=='conflicting_head' for a in r['associations']))
    def test_no_motorcycle(self):
        self.assertEqual(associate([p(1,[0,0,20,20])])['associations'][0]['status'],'unassigned')
if __name__=='__main__':unittest.main()
