"""Passive hashes for paired training reproducibility diagnostics."""
import hashlib
import random
import numpy as np
import torch


def tensors_hash(tensors):
    digest=hashlib.sha256()
    for value in tensors:
        if value is None:
            digest.update(b'None');continue
        value=value.detach().contiguous().cpu()
        digest.update(str((str(value.dtype),tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def rng_hashes():
    return {'torch_cpu':tensors_hash([torch.get_rng_state()]),
            'torch_cuda':tensors_hash(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None,
            'python':hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            'numpy':hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest()}


class SamplerTrace:
    def __init__(self,original,records,stage):
        self.original=original;self.records=records;self.stage=stage
        self.batch_size_per_image=original.batch_size_per_image
        self.positive_fraction=original.positive_fraction

    def __call__(self,labels):
        positive,negative=self.original(labels)
        self.records.append({'stage':self.stage,'labels_sha256':tensors_hash(labels),
                             'positive_sha256':tensors_hash(positive),'negative_sha256':tensors_hash(negative),
                             'positive_counts':[int(v.sum()) for v in positive],
                             'negative_counts':[int(v.sum()) for v in negative]})
        return positive,negative
