"""Adapt visual features to all eligible project photos using rotation prediction."""
import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomResizedCrop, ToTensor, Normalize, Compose
from data import read_jsonl


class ProjectPhotos(Dataset):
    def __init__(self, folder):
        rows = read_jsonl("reports/preparation/image-decisions.jsonl")
        def split(row):
            return row["split"] or row["upstreamSplit"]
        held = [r for r in rows if split(r) != "train"]
        held_groups = {r["groupId"] for r in held}
        held_hashes = {r["sha256"] for r in held}
        seen = set()
        self.rows = []
        for r in rows:
            if split(r) == "train" and r["groupId"] not in held_groups and r["sha256"] not in held_hashes and r["sha256"] not in seen:
                self.rows.append(r)
                seen.add(r["sha256"])
        (folder/"photos.jsonl").write_text("".join(json.dumps(r)+"\n" for r in self.rows))
        counts = Counter(r["source"] for r in self.rows)
        (folder/"sources.json").write_text(json.dumps(counts,indent=2))
        self.transform = Compose([RandomResizedCrop(160,scale=(.65,1.)),ToTensor(),Normalize((.5,)*3,(.5,)*3)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        with Image.open(self.rows[index]["image"]) as photo:
            x=self.transform(ImageOps.exif_transpose(photo).convert("RGB"))
        k=random.randrange(4)
        return torch.rot90(x,k,(1,2)),k


def adapt(model,folder):
    data = ProjectPhotos(folder)
    loader = DataLoader(data,batch_size=64,num_workers=3,persistent_workers=True,shuffle=True,pin_memory=True)
    head=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((2,2)),torch.nn.Flatten(),torch.nn.Linear(512,4)).cuda()
    optimizer=torch.optim.AdamW([{"params":model.backbone.parameters(),"lr":.00005},{"params":head.parameters(),"lr":.001}])
    for epoch in range(2):
        correct = 0
        for images,labels in loader:
            optimizer.zero_grad(set_to_none=True)
            labels=labels.cuda(non_blocking=True)
            pred=head(model.backbone(images.cuda(non_blocking=True)))
            torch.nn.functional.cross_entropy(pred,labels).backward()
            optimizer.step()
            correct+=(pred.argmax(1)==labels).sum().item()
        print("rotation",epoch+1,len(data),correct/len(data),flush=True)
