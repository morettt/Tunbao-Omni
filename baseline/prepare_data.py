"""Validate prepared 20fps HumanML3D features and preserve caption/group mappings."""
import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from paths import DATA_ROOT

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--captions', type=Path, required=True, help='JSONL: id, group, captions (list of English strings)')
    args = parser.parse_args()
    output = DATA_ROOT / 'features'
    mapping = output / 'motion_text_mapping.jsonl'
    if mapping.exists():
        raise FileExistsError('Prepared mapping already exists: ' + str(mapping))
    arrays = output / 'new_joint_vecs'
    arrays.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for line in args.captions.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key, group, captions = item['id'], item['group'], item['captions']
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', key) or key in seen:
            raise ValueError('Invalid or duplicate motion ID: ' + str(key))
        if not isinstance(group, str) or not group:
            raise ValueError('group must identify the original capture take')
        if not isinstance(captions, list) or not captions or not all(isinstance(x, str) and x.strip() for x in captions):
            raise ValueError('Each motion needs at least one nonempty English caption')
        source = (args.features / (key + '.npy')).resolve()
        value = np.load(source, allow_pickle=False)
        if value.ndim != 2 or value.shape[1] != 263 or len(value) < 4 or not np.isfinite(value).all():
            raise ValueError('Expected finite [frames,263] motion features: ' + key)
        destination = arrays / (key + '.npy')
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
                raise ValueError('An existing feature file differs: ' + key)
        else:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        rows.append(dict(id=key, frames=len(value), feature_file='new_joint_vecs/' + key + '.npy',
            feature_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            source_path=source.name, selection_index=len(rows),
            descriptions={f'caption_{i+1}': text for i, text in enumerate(captions)},
            source_metadata=dict(take_name=group, actor_uid=item.get('actor', 'unspecified'))))
        seen.add(key)
    if not rows:
        raise ValueError('No motions were supplied')
    split_groups = {name:set() for name in ['train', 'val']}
    eligible = {name:0 for name in split_groups}
    for row in rows:
        group = row['source_metadata']['take_name']
        split = 'val' if int(hashlib.sha256(('123:' + group).encode()).hexdigest()[:8], 16) % 10 == 0 else 'train'
        split_groups[split].add(group)
        eligible[split] += int(row['frames'] >= 64)
    if not all(eligible.values()):
        raise ValueError('Both train and validation need >=64-frame motions from distinct take groups')
    temporary = mapping.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    temporary.replace(mapping)
    print(json.dumps(dict(motions=len(rows), eligible_vq_motions=eligible, shared_take_groups=0)))

if __name__ == '__main__':
    main()
