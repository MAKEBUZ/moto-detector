import random
import unittest
import torch
from data import SeparateHelmetDataset


class PhotometricTests(unittest.TestCase):
    def test_geometry_and_flip_rng_preserved(self):
        root='data/prepared/helmet-separate-v2-expanded-01'
        base=SeparateHelmetDataset(root,'train')
        photo=SeparateHelmetDataset(root,'train',photometric=True)
        random.seed(42); image,a=base[0]; state=random.getstate()
        random.seed(42); changed,b=photo[0]
        self.assertEqual(state,random.getstate())
        self.assertTrue(torch.equal(a['boxes'],b['boxes']))
        self.assertTrue(torch.equal(a['labels'],b['labels']))
        self.assertFalse(torch.equal(image,changed))
        self.assertEqual(image.shape,changed.shape)
        val=SeparateHelmetDataset(root,'val',photometric=True)
        self.assertFalse(val.photometric)
        self.assertTrue(torch.equal(val[0][0],val[0][0]))


if __name__=='__main__':unittest.main()
