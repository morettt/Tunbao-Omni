from paths import DATA_ROOT, T2M_ROOT, CLIP_PATH
"""Freeze best  VQ and build a resumable, split-preserving motion token cache."""
import os,sys,json,time,hashlib,shutil,sqlite3,fcntl
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('OMP_NUM_THREADS','4')
ROOT=DATA_ROOT;PROJECT=T2M_ROOT
sys.path[:0]=[str(PROJECT)]
import numpy as np
import torch
from models.vqvae import HumanVQVAE
from common import atomic_json,sha256

def main():
    torch.set_num_threads(4);torch.manual_seed(123);torch.backends.cuda.matmul.allow_tf32=False
    run=ROOT/'vq';data=ROOT/'features';out=ROOT/'tokens'
    out.mkdir(exist_ok=True)
    lock=(out/'.encoding.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'tokens').mkdir(exist_ok=True);(out/'meta').mkdir(exist_ok=True)
    expected=sha256(run/'net_best_fid.pth')
    source=run/'net_best_fid.pth';assert sha256(source)==expected
    for original,target in [(source,out/'vq_frozen.pth'),(run/'meta/mean.npy',out/'meta/mean.npy'),(run/'meta/std.npy',out/'meta/std.npy')]:
        if not target.exists():
            tmp=target.with_suffix(target.suffix+'.tmp');shutil.copy2(original,tmp);tmp.replace(target)
        assert sha256(original)==sha256(target)
    rows=[json.loads(line) for line in (data/'motion_text_mapping.jsonl').read_text().splitlines()]
    splits={name:set((run/(name+'_all.txt')).read_text().splitlines()) for name in ['train','val']}
    assert not splits['train']&splits['val']
    assert splits['train']|splits['val']=={row['id'] for row in rows}
    takes={name:{row['source_metadata']['take_name'] for row in rows if row['id'] in names} for name,names in splits.items()}
    assert not takes['train']&takes['val']
    saved_step=int(torch.load(source,map_location='cpu',weights_only=False)['iteration'])
    cfg=dict(checkpoint_step=saved_step,checkpoint_sha256=expected,source_mapping_sha256=sha256(data/'motion_text_mapping.jsonl'),
        normalization_mean_sha256=sha256(out/'meta/mean.npy'),normalization_std_sha256=sha256(out/'meta/std.npy'),
        codebook_size=512,frames_per_token=4,fps=20,token_dtype='uint16',seed=123,
        cropping='full motion, trim only 0..3 final frames to a multiple of four',long_sequences='preserved without semantic slicing',
        split='original take_name groups, includes short clips omitted only from VQ training',reserved_tokens='EOS/PAD added by future generator loader, not stored here')
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==cfg
    else:atomic_json(out/'config.json',cfg)
    checkpoint=torch.load(out/'vq_frozen.pth',map_location='cpu',weights_only=False);assert checkpoint['iteration']==saved_step
    model=HumanVQVAE(SimpleNamespace(dataname='t2m',quantizer='ema_reset',mu=.99),512,512,512,2,2,512,3,3,'relu',None).cuda().eval()
    model.load_state_dict(checkpoint['net'],strict=True);model.requires_grad_(False);del checkpoint
    mean=np.load(out/'meta/mean.npy');std=np.load(out/'meta/std.npy')
    db=sqlite3.connect(out/'checkpoint.sqlite3');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS tokens (id TEXT PRIMARY KEY, metadata TEXT NOT NULL)');db.commit()
    completed={key for key, in db.execute('SELECT id FROM tokens')}
    assert completed<={r['id'] for r in rows}
    started=time.monotonic();new=0;total_tokens=0;counts=np.zeros(512,dtype=np.int64)
    def report(stage,**extra):
        rate=new/max(time.monotonic()-started,1)
        atomic_json(out/'status.json',dict(stage=stage,pid=os.getpid(),encoded=len(completed),total=len(rows),
            elapsed_seconds=round(time.monotonic()-started,1),eta_seconds=round((len(rows)-len(completed))/rate,1) if rate else None,
            updated=time.strftime('%Y-%m-%d %H:%M:%S'),**extra))
    report('encoding');print('ENCODING_START',len(rows),'RESUMED',len(completed),flush=True)
    with torch.inference_mode():
        for row in rows:
            key=row['id']
            if key in completed:continue
            path=data/row['feature_file'];assert sha256(path)==row['feature_sha256']
            features=np.load(path);length=len(features)//4*4
            assert length>=4 and features.shape[1]==263 and np.isfinite(features).all()
            x=torch.from_numpy(((features[:length]-mean)/std).astype(np.float32)).unsqueeze(0).cuda()
            tokens=model.encode(x).cpu().numpy()
            assert tokens.shape==(1,length//4) and tokens.min()>=0 and tokens.max()<512
            tokens=tokens[0].astype(np.uint16)
            token_path=out/'tokens'/(key+'.npy');tmp=token_path.with_suffix('.npy.tmp')
            with tmp.open('wb') as f:np.save(f,tokens)
            tmp.replace(token_path)
            split='train' if key in splits['train'] else 'val'
            record=dict(name=key,source_motion_id=key,split=split,source_path=row['source_path'],
                original_frames=len(features),encoded_frames=length,trimmed_tail_frames=len(features)-length,
                tokens=len(tokens),token_file='tokens/'+key+'.npy',token_sha256=sha256(token_path),source_feature_sha256=row['feature_sha256'],
                caption_fields=row['descriptions'],captions=list(row['descriptions'].values()),selection_index=row['selection_index'],
                take_name=row['source_metadata']['take_name'],actor_uid=row['source_metadata']['actor_uid'])
            assert record['captions']
            db.execute('INSERT INTO tokens VALUES(?,?)',(key,json.dumps(record,ensure_ascii=False)));db.commit();completed.add(key);new+=1
            if len(completed)%100==0:report('encoding');print('ENCODING_PROGRESS',len(completed),'/',len(rows),flush=True)
    report('validating')
    records={key:json.loads(value) for key,value in db.execute('SELECT id,metadata FROM tokens')}
    assert set(records)=={r['id'] for r in rows}
    handles={name:(out/(name+'.jsonl.tmp')).open('w',encoding='utf-8') for name in splits}
    summary={name:dict(motions=0,captions=0,tokens=0,encoded_frames=0,original_frames=0,trimmed_tail_frames=0,max_tokens=0,over_49_tokens=0) for name in splits}
    digest=hashlib.sha256()
    try:
        for row in rows:
            record=records[row['id']];path=out/record['token_file'];tokens=np.load(path)
            assert np.issubdtype(tokens.dtype,np.integer) and tokens.ndim==1 and len(tokens)==record['tokens']
            assert len(tokens)*4==record['encoded_frames'] and 0<=record['trimmed_tail_frames']<=3
            assert tokens.min()>=0 and tokens.max()<512 and sha256(path)==record['token_sha256']
            assert record['caption_fields']==row['descriptions'] and record['source_feature_sha256']==row['feature_sha256']
            assert record['name'] in splits[record['split']] and record['selection_index']==row['selection_index']
            # Mirror the existing generator JSONL format without EOS or padding.
            sample={**record,'token_variants':[tokens.astype(int).tolist()]}
            handles[record['split']].write(json.dumps(sample,ensure_ascii=False)+'\n')
            s=summary[record['split']];s['motions']+=1;s['captions']+=len(record['captions']);s['tokens']+=len(tokens)
            for k in ['encoded_frames','original_frames','trimmed_tail_frames']:s[k]+=record[k]
            s['max_tokens']=max(s['max_tokens'],len(tokens));s['over_49_tokens']+=int(len(tokens)>49)
            counts+=np.bincount(tokens,minlength=512);total_tokens+=len(tokens)
            digest.update((record['name']+':'+record['token_sha256']+'\n').encode())
    finally:
        for f in handles.values():f.close()
    for name in splits:(out/(name+'.jsonl.tmp')).replace(out/(name+'.jsonl'))
    assert sha256(out/'vq_frozen.pth')==expected
    assert sha256(run/'meta/mean.npy')==cfg['normalization_mean_sha256'] and sha256(run/'meta/std.npy')==cfg['normalization_std_sha256']
    for s in summary.values():s['encoded_hours']=s['encoded_frames']/72000
    manifest=dict(stage='tokens_complete',checkpoint_step=28000,checkpoint_sha256=expected,motions=len(rows),
        total_tokens=total_tokens,used_codes=int((counts>0).sum()),code_counts=counts.tolist(),splits=summary,
        same_take_leakage=0,text_mapping_mismatches=0,token_files_sha256=digest.hexdigest(),
        token_files_bytes=sum(p.stat().st_size for p in (out/'tokens').glob('*.npy')),
        jsonl_bytes=sum((out/(name+'.jsonl')).stat().st_size for name in splits),finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    atomic_json(out/'complete.json',manifest);report('tokens_complete',summary=summary,total_tokens=total_tokens)
    print('ENCODING_COMPLETE',json.dumps({k:v for k,v in manifest.items() if k!='code_counts'}),flush=True)

if __name__=='__main__':main()
