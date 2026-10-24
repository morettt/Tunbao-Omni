from paths import DATA_ROOT, T2M_ROOT, CLIP_PATH
"""Filter >49 motion tokens; cache frozen CLIP features and evaluator captions."""
import os,sys,json,hashlib,time,fcntl
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
ROOT=DATA_ROOT;PROJECT=T2M_ROOT;CACHE=ROOT/'tokens';OUT=ROOT/'gpt'
sys.path[:0]=[str(PROJECT)]
import numpy as np
import torch,clip,spacy
from common import sha256,atomic_json
from utils.word_vectorizer import WordVectorizer

def main():
    torch.set_num_threads(4);torch.manual_seed(123);OUT.mkdir(exist_ok=True)
    lock=(OUT/'.prepare.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    expected=sha256(CACHE/'complete.json')
    if (OUT/'prepared.json').exists():
        assert json.loads((OUT/'prepared.json').read_text())['source_cache_sha256']==expected
        print('GPT_DATA_ALREADY_PREPARED',flush=True);return
    records={};report={};captions=[];caption_ids={}
    for split in ['train','val']:
        source=[json.loads(s) for s in (CACHE/(split+'.jsonl')).read_text().splitlines()]
        retained=[x for x in source if 1<=x['tokens']<=49];records[split]=retained
        report[split]=dict(motions=len(retained),excluded_long=sum(x['tokens']>49 for x in source),captions=sum(len(x['captions']) for x in retained),
            encoded_hours=sum(x['encoded_frames'] for x in retained)/72000,min_tokens=min(x['tokens'] for x in retained),max_tokens=max(x['tokens'] for x in retained))
        for row in retained:
            tokens=np.load(CACHE/row['token_file']);assert len(tokens)==row['tokens'] and 1<=len(tokens)<=49 and tokens.min()>=0 and tokens.max()<512
            assert sha256(CACHE/row['token_file'])==row['token_sha256']
            ids=[]
            for caption in row['captions']:
                if caption not in caption_ids:caption_ids[caption]=len(captions);captions.append(caption)
                ids.append(caption_ids[caption])
            row['caption_feature_ids']=ids
    assert len(records['train'])>0 and len(records['val'])>0
    assert not {x['take_name'] for x in records['train']}&{x['take_name'] for x in records['val']}
    model,_=clip.load(str(CLIP_PATH),device='cuda',jit=False);model.eval().requires_grad_(False)
    features=np.empty((len(captions),512),np.float32);truncated=0
    with torch.inference_mode():
        for start in range(0,len(captions),256):
            texts=captions[start:start+256]
            for text in texts:truncated+=int(len(clip.clip._tokenizer.encode(text))+2>77)
            features[start:start+len(texts)]=model.encode_text(clip.tokenize(texts,truncate=True).cuda()).float().cpu().numpy()
            if start%2560==0:print('CLIP_FEATURES',start,'/',len(captions),flush=True)
    assert np.isfinite(features).all();np.save(OUT/'clip_features.npy',features)
    atomic_json(OUT/'captions.json',captions)
    nlp=spacy.load('en_core_web_sm',disable=['parser','ner']);vectorizer=WordVectorizer(str(PROJECT/'glove'),'our_vab')
    words=[];poses=[];lengths=[]
    # A fixed caption per held-out motion gives reproducible validation pairs.
    for row in records['val']:
        choice=int(hashlib.sha256(('123:'+row['name']).encode()).hexdigest()[:8],16)%len(row['captions'])
        row['validation_caption_index']=choice;text=row['captions'][choice].replace('-','')
        tags=[]
        for token in nlp(text):
            if not token.text.isalpha():continue
            word=token.lemma_ if token.pos_ in ['NOUN','VERB'] and token.text!='left' else token.text
            tags.append(word+'/'+token.pos_)
        tags=['sos/OTHER']+tags[:20]+['eos/OTHER'];lengths.append(len(tags));tags+=['unk/OTHER']*(22-len(tags))
        pairs=[vectorizer[t] for t in tags];words.append(np.stack([a for a,b in pairs]));poses.append(np.stack([b for a,b in pairs]))
    np.savez(OUT/'validation_text.npz',words=np.array(words,dtype=np.float32),poses=np.array(poses,dtype=np.float32),lengths=np.array(lengths,dtype=np.int64))
    for split,rows in records.items():
        temp=OUT/(split+'.jsonl.tmp');temp.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8');temp.replace(OUT/(split+'.jsonl'))
    atomic_json(OUT/'prepared.json',dict(source_cache_sha256=expected,vq_sha256=sha256(CACHE/'vq_frozen.pth'),splits=report,
        unique_captions=len(captions),clip_truncated_captions=truncated,clip_sha256=sha256(CLIP_PATH),
        clip_features_sha256=sha256(OUT/'clip_features.npy'),train_sha256=sha256(OUT/'train.jsonl'),val_sha256=sha256(OUT/'val.jsonl'),
        filtering='only discard sequences with >49 motion tokens; no clipping or resplitting',shared_takes=0,spacy_version=spacy.__version__,finished=time.time()))
    print('GPT_DATA_READY',json.dumps(report),flush=True)

if __name__=='__main__':main()
