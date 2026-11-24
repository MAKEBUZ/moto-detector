import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from torchvision.transforms.functional import pil_to_tensor


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


class HelmetDataset(Dataset):
    def __init__(self, root, task, split, limit=None, augment=True):
        self.root, self.task = Path(root), task
        self.augment = augment and split == 'train'
        manifest = self.root / f'{task}-manifest.jsonl'
        self.rows = [r for r in read_jsonl(manifest) if r['split'] == split]
        if limit:
            self.rows = random.Random(20261123).sample(self.rows, min(limit, len(self.rows)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        if self.task == 'classifier':
            filename = self.root / 'classifier' / r['split'] / r['className'] / (r['id'] + '.jpg')
        else:
            filename = self.root / r['image'] if getattr(self, 'separate', False) else self.root / 'detector' / 'images' / r['split'] / (r['id'] + '.jpg')
        with Image.open(filename) as im:
            image = im.convert('RGB')
        if self.task == 'classifier':
            image = ImageOps.pad(image, (96, 96), method=Image.Resampling.BILINEAR, color=(127, 127, 127))
            if self.augment:
                if random.random() < .5:
                    image = ImageOps.mirror(image)
                image = ImageEnhance.Brightness(image).enhance(random.uniform(.85, 1.15))
            return pil_to_tensor(image).float().div(255).sub(.5).div(.5), r['classId']
        old_w, old_h = image.size
        scale = min(1., 640 / max(old_w, old_h))
        w, h = max(1, round(old_w * scale)), max(1, round(old_h * scale))
        image = image.resize((w, h), Image.Resampling.BILINEAR)
        boxes = torch.tensor([a['box'] for a in r['annotations']], dtype=torch.float32)
        boxes *= torch.tensor([w / old_w, h / old_h, w / old_w, h / old_h])
        regions = None
        if getattr(self, 'exposure_regions', False):
            regions = torch.tensor([n['bbox'] for n in r.get('reviewed_nonhead_regions', [])], dtype=torch.float32).reshape(-1, 4)
            regions *= torch.tensor([w / old_w, h / old_h, w / old_w, h / old_h])
        if self.augment and random.random() < .5:
            image = ImageOps.mirror(image)
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if regions is not None:
                regions[:, [0, 2]] = w - regions[:, [2, 0]]
        if getattr(self, 'photometric', False):
            image = ImageEnhance.Brightness(image).enhance(self.photo_rng.uniform(.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(self.photo_rng.uniform(.85, 1.15))
            if self.photo_rng.random() < .25:
                image = image.filter(ImageFilter.GaussianBlur(radius=self.photo_rng.uniform(.2, .8)))
        labels = torch.tensor([a['classId'] + 1 for a in r['annotations']], dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor(index)}
        if regions is not None: target['negative_regions'] = regions
        return pil_to_tensor(image).float().div(255), target


def collate_detection(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class SeparateHelmetDataset(HelmetDataset):
    """Four-class reviewed manifests; paths relative to the dataset root."""
    def __init__(self, root, split, limit=None, augment=True, photometric=False, photo_seed=20261124):
        self.root, self.task, self.separate = Path(root), 'detector', True
        if split not in ('train', 'val'):
            raise ValueError('Only train/val supported')
        self.augment = augment and split == 'train'
        self.photometric = photometric and self.augment
        # Separate stream preserves the baseline horizontal-flip RNG sequence.
        self.photo_rng = random.Random(photo_seed)
        self.rows = read_jsonl(self.root / f'{split}-manifest.jsonl')
        for i, row in enumerate(self.rows):
            if row['split'] != split:
                raise ValueError('Split mismatch')
            image_path = (self.root / row['image']).resolve()
            if not image_path.is_relative_to(self.root.resolve()):
                raise ValueError('Image outside dataset')
            if not row['annotations'] or any(a['classId'] not in range(4) for a in row['annotations']):
                raise ValueError('Invalid four-class annotation')
            row['id'] = i
        if limit:
            self.rows = random.Random(20261123).sample(self.rows, min(limit, len(self.rows)))
