from paths import DATA_ROOT, T2M_ROOT, CLIP_PATH
"""Freeze take-disjoint splits and fit normalization on eligible training motions."""
from pathlib import Path
import json,hashlib
import numpy as np

ROOT=DATA_ROOT
DATA=ROOT/'features';OUT=ROOT/'vq'
OUT.mkdir(exist_ok=True);META=OUT/'meta';META.mkdir(exist_ok=True)
source=DATA/'motion_text_mapping.jsonl'
rows=[json.loads(line) for line in source.read_text().splitlines()]
split=lambda row:'val' if int(hashlib.sha256(('123:'+row['source_metadata']['take_name']).encode()).hexdigest()[:8],16)%10==0 else 'train'
eligible={name:[row for row in rows if split(row)==name and row['frames']>=64] for name in ['train','val']}
groups={name:{row['source_metadata']['take_name'] for row in rows if split(row)==name} for name in ['train','val']}
assert not groups['train']&groups['val']
if (OUT/'dataset.json').exists():
    previous=json.loads((OUT/'dataset.json').read_text())
    assert previous['source_mapping_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    print(json.dumps(previous));raise SystemExit
for name,items in eligible.items():
    (OUT/(name+'.txt')).write_text('\n'.join(row['id'] for row in items)+'\n')
    (OUT/(name+'_all.txt')).write_text('\n'.join(row['id'] for row in rows if split(row)==name)+'\n')
sum1=np.zeros(263);sum2=np.zeros(263);frames=0
for row in eligible['train']:
    path=DATA/row['feature_file'];x=np.load(path).astype(np.float64)
    assert x.shape==(row['frames'],263) and np.isfinite(x).all()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['feature_sha256']
    sum1+=x.sum(0);sum2+=(x*x).sum(0);frames+=len(x)
mean=sum1/frames;std=np.sqrt(np.maximum(sum2/frames-mean*mean,0))
# Same semantic grouping as HumanML3D cal_mean_variance.ipynb.
for start,end in [(0,1),(1,3),(3,4),(4,67),(67,193),(193,259),(259,263)]:std[start:end]=max(float(std[start:end].mean()),1e-6)
np.save(META/'mean.npy',mean.astype(np.float32));np.save(META/'std.npy',std.astype(np.float32))
report=dict(source_mapping_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),selected_clips=len(rows),selected_hours=sum(r['frames'] for r in rows)/72000,
    training_clips=len(eligible['train']),training_hours=frames/72000,validation_clips=len(eligible['val']),validation_hours=sum(r['frames'] for r in eligible['val'])/72000,
    excluded_short_clips=sum(r['frames']<64 for r in rows),excluded_short_hours=sum(r['frames'] for r in rows if r['frames']<64)/72000,
    split_group='take_name',split_seed=123,shared_take_groups=0,window_size=64,normalization_fit='eligible_training_only',
    mean_sha256=hashlib.sha256((META/'mean.npy').read_bytes()).hexdigest(),std_sha256=hashlib.sha256((META/'std.npy').read_bytes()).hexdigest())
(OUT/'dataset.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
