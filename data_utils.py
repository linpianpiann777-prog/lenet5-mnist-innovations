"""Offline, checksum-validated IDX loading. Never downloads implicitly."""
from pathlib import Path
import gzip
import hashlib
import struct
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'MNIST' / 'raw'
CLASSES = [str(i) for i in range(10)]
CLASSES_ZH = CLASSES
MD5 = {'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
       'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
       't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
       't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c'}



def load_part(train=True):
    prefix = 'train' if train else 't10k'
    arrays = []
    for kind in ('images-idx3', 'labels-idx1'):
        name = f'{prefix}-{kind}-ubyte.gz'
        path = RAW / name
        if not path.exists():
            raise FileNotFoundError(f'请按 README 下载数据文件: {path}')
        content = path.read_bytes()
        if hashlib.md5(content).hexdigest() != MD5[name]:
            raise ValueError(f'MD5 mismatch: {name}')
        content = gzip.decompress(content)
        if kind.startswith('images'):
            magic, n, h, w = struct.unpack('>IIII', content[:16])
            assert magic == 2051 and (h, w) == (28, 28)
            arrays.append(np.frombuffer(content, np.uint8, offset=16).reshape(n, h, w).copy())
        else:
            magic, n = struct.unpack('>II', content[:8])
            assert magic == 2049
            arrays.append(np.frombuffer(content, np.uint8, offset=8).copy())
        assert len(arrays[-1]) == n
    return arrays


def split_indices(labels, seed=2026):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(labels == c))
        val.extend(idx[:600]); train.extend(idx[600:])
    return np.array(train, dtype=np.int64), np.array(val, dtype=np.int64)


def tensor_images(images, device):
    x = torch.as_tensor(images, device=device).unsqueeze(1).float().div_(255).sub_(0.5).div_(0.5)
    return torch.nn.functional.pad(x, (2, 2, 2, 2), value=-1)
