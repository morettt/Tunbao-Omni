from paths import DATA_ROOT, T2M_ROOT, CLIP_PATH
"""Checkpoint and deterministic batch-stream support for the isolated reproduction."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    os.replace(tmp, path)


def atomic_save(path, value, rotate=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    if rotate and path.exists():
        os.replace(path, path.with_name(path.stem + '_previous' + path.suffix))
    os.replace(tmp, path)


class SeededDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        index, sample_seed = key
        # Upstream datasets use Python / NumPy for crops and token augmentation.
        random.seed(sample_seed)
        np.random.seed(sample_seed % (2**32))
        return self.dataset[index]


class ResumableBatches:
    def __init__(self, dataset_len, batch_size, seed, consumed=0):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.seed = seed
        self.consumed = consumed
        self.per_epoch = dataset_len // batch_size
        if not self.per_epoch:
            raise ValueError('Dataset smaller than one full batch')

    def __iter__(self):
        batch_number = self.consumed
        cached_epoch, order = -1, None
        while True:
            epoch, offset = divmod(batch_number, self.per_epoch)
            if epoch != cached_epoch:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                order = torch.randperm(self.dataset_len, generator=generator).tolist()
                cached_epoch = epoch
            indices = order[offset * self.batch_size:(offset + 1) * self.batch_size]
            yield [(idx, (self.seed + epoch * self.dataset_len + idx) % (2**32)) for idx in indices]
            batch_number += 1


class BatchStream:
    def __init__(self, dataset, batch_size, seed, workers=8, consumed=0):
        self.sampler = ResumableBatches(len(dataset), batch_size, seed, consumed)
        self.loader = DataLoader(SeededDataset(dataset), batch_sampler=self.sampler,
                                 num_workers=workers, pin_memory=True,
                                 generator=torch.Generator().manual_seed(seed + 9876))
        self.iterator = None

    def next(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        batch = next(self.iterator)
        self.sampler.consumed += 1
        return batch

    def state_dict(self):
        return {'batches_consumed': self.sampler.consumed,
                'batch_size': self.sampler.batch_size,
                'dataset_len': self.sampler.dataset_len,
                'seed': self.sampler.seed}

    def load_state_dict(self, state):
        assert self.iterator is None, 'Restore before iterating'
        for name in ['batch_size', 'dataset_len', 'seed']:
            if state[name] != getattr(self.sampler, name):
                raise ValueError('Data-stream configuration changed: ' + name)
        self.sampler.consumed = state['batches_consumed']


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def quantizer_state(model):
    result = {}
    for name, module in model.named_modules():
        if hasattr(module, 'codebook') and hasattr(module, 'init'):
            result[name] = {key: getattr(module, key) for key in ['init', 'code_sum', 'code_count']}
    return result


def restore_quantizer(model, state):
    modules = dict(model.named_modules())
    for name, extra in state.items():
        module = modules[name]
        device = module.codebook.device
        for key, value in extra.items():
            setattr(module, key, value.to(device) if torch.is_tensor(value) else value)


def checkpoint(model, optimizer, scheduler, scaler, stream, **metadata):
    return dict(schema_version=1, model=model.state_dict(), quantizer=quantizer_state(model),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(), data=stream.state_dict(), rng=rng_state(), **metadata)


def restore(path, model, optimizer, scheduler, scaler, stream):
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value.get('schema_version') != 1:
        raise ValueError('A full reproduction checkpoint is required')
    model.load_state_dict(value['model'], strict=True)
    restore_quantizer(model, value['quantizer'])
    optimizer.load_state_dict(value['optimizer'])
    scheduler.load_state_dict(value['scheduler'])
    scaler.load_state_dict(value['scaler'])
    stream.load_state_dict(value['data'])
    restore_rng(value['rng'])
    return value


def autocast(precision):
    if precision == 'fp32':
        return contextlib.nullcontext()
    return torch.autocast('cuda', dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)


def setup_precision(model, precision):
    if precision != 'fp32':
        # EMA accumulation and nearest-code distance stay in FP32 under AMP.
        for module in model.modules():
            if hasattr(module, 'codebook') and hasattr(module, 'init'):
                original = module.forward
                def forward(x, original=original):
                    with torch.autocast('cuda', enabled=False):
                        return original(x.float())
                module.forward = forward
    return torch.amp.GradScaler('cuda', enabled=precision == 'fp16')


def update(loss, model, optimizer, scaler):
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite training loss')
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return float(grad_norm)


def environment():
    return {'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0),
            'gpu_memory_gib': torch.cuda.get_device_properties(0).total_memory / 2**30,
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32}
