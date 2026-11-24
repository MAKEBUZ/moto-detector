import copy
import unittest
from validate_annotation_batch import validate


class NegativeReviewTests(unittest.TestCase):
    def setUp(self):
        region={'id':'n0','bbox':[1,1,4,4],'predicted_label':4,'score':.8,'checkpoint':'own','checkpoint_sha256':'hash','decision':'pending','category':'pending'}
        self.ref={'schema_version':1,'batch_id':'example','training_ready':False,'scenes':[{'image_id':'image','image':'unused','sha256':'hash','width':10,'height':10,'split':'train','group_id':'group','entities':[],'links':[],'hard_negative_candidates':[region]}]}
        self.review=copy.deepcopy(self.ref);self.review.update(human_reviewed=True,reviewer='unit-test',reviewed_at='2026-11-24T00:00:00+00:00')
        self.review['scenes'][0].update(reviewed=True,annotation_complete=True)

    def check_review(self):return validate(self.review,self.ref,reviewed=True,verify_images=False)

    def test_pending_cannot_pass(self):self.assertFalse(self.check_review()['valid'])

    def test_confirmed_background_needs_category(self):
        r=self.review['scenes'][0]['hard_negative_candidates'][0];r['decision']='background'
        self.assertFalse(self.check_review()['valid']);r['category']='seat';self.assertTrue(self.check_review()['valid'])

    def test_uncertain_cannot_claim_complete(self):
        self.review['scenes'][0]['hard_negative_candidates'][0]['decision']='uncertain'
        self.assertFalse(self.check_review()['valid']);self.review['scenes'][0]['annotation_complete']=False;self.assertTrue(self.check_review()['valid'])

    def test_candidate_cannot_be_removed_or_moved(self):
        r=self.review['scenes'][0]['hard_negative_candidates'][0];r.update(decision='background',category='seat');r['bbox']=[2,2,5,5]
        self.assertFalse(self.check_review()['valid']);self.review['scenes'][0]['hard_negative_candidates']=[];self.assertFalse(self.check_review()['valid'])


if __name__=='__main__':unittest.main()
