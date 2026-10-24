from paths import DATA_ROOT, T2M_ROOT, CLIP_PATH
"""Original T2M-GPT architecture on <=49-token custom-motion motions, full resume state."""
import os,sys,time,json,signal,fcntl,argparse,random
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MPLBACKEND','Agg')
ROOT=DATA_ROOT;PROJECT=T2M_ROOT;sys.path[:0]=[str(PROJECT)];os.chdir(PROJECT)
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from models.t2m_trans import Text2Motion_Transformer
from models.vqvae import HumanVQVAE
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.utils_model import initial_optim
from common import seed_all,BatchStream,checkpoint,restore,atomic_save,atomic_json,setup_precision,update,rng_state,restore_rng,sha256,environment
from cached_gpt_eval import CachedGPT
from train_vq import fid

class MotionTexts(Dataset):
    def __init__(self,rows,features):self.rows,self.features=rows,features
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];codes=np.array(random.choice(row['token_variants']),dtype=np.int64)
        if len(codes)>1 and np.random.choice([False,False,True]):codes=codes[1:] if random.choice([False,True]) else codes[:-1]
        target=np.full(51,513,dtype=np.int64);target[:len(codes)]=codes;target[len(codes)]=512
        feature=self.features[random.choice(row['caption_feature_ids'])]
        return feature.copy(),target,len(codes)

@torch.inference_mode()
def sample(model,features):
    cache=CachedGPT(model,max_batch=len(features));codes=[];lengths=torch.full((len(features),),-1,device='cuda',dtype=torch.long);last=None
    for step in range(50):
        logits=cache.step(last,features,step);assert torch.isfinite(logits).all()
        nxt=torch.distributions.Categorical(logits=logits).sample();lengths[(lengths<0)&(nxt==512)]=step;codes.append(nxt);last=nxt
        if (lengths>=0).all():break
    sequence=torch.stack(codes,dim=1);lengths[lengths<0]=49;fallback=int((lengths==0).sum())
    result=[sequence[i:i+1,:int(n)] if int(n)>0 else torch.ones((1,1),device='cuda',dtype=torch.long) for i,n in enumerate(lengths)]
    assert all(1<=x.shape[1]<=49 and x.min()>=0 and x.max()<512 for x in result)
    return result,fallback

def retrieval(text,motion):
    distance=np.sqrt(np.maximum((text*text).sum(1)[:,None]+(motion*motion).sum(1)[None,:]-2*text@motion.T,0))
    order=np.argsort(distance,axis=1);correct=order==np.arange(len(text))[:,None]
    return np.array([correct[:,:k].any(1).sum() for k in [1,2,3]]),float(np.trace(distance))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=300000);parser.add_argument('--resume');parser.add_argument('--eval-every',type=int,default=1000);parser.add_argument('--eval-now',action='store_true')
    args=parser.parse_args();out=ROOT/'gpt';cache=ROOT/'tokens'
    lock=(out/'.training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'config.json').exists() and not args.resume:raise RuntimeError('Existing run requires explicit resume')
    prepared=json.loads((out/'prepared.json').read_text());assert sha256(cache/'vq_frozen.pth')==prepared['vq_sha256']
    assert sha256(out/'clip_features.npy')==prepared['clip_features_sha256']
    train=[json.loads(x) for x in (out/'train.jsonl').read_text().splitlines()];val=[json.loads(x) for x in (out/'val.jsonl').read_text().splitlines()]
    assert sha256(out/'train.jsonl')==prepared['train_sha256'] and sha256(out/'val.jsonl')==prepared['val_sha256']
    assert len(train)==prepared['splits']['train']['motions'] and len(val)==prepared['splits']['val']['motions'] and all(1<=r['tokens']<=49 for r in train+val)
    features=np.load(out/'clip_features.npy');text=np.load(out/'validation_text.npz');seed_all(123);torch.backends.cuda.matmul.allow_tf32=False
    ds=MotionTexts(train,features);stream=BatchStream(ds,128,123,workers=4)
    model=Text2Motion_Transformer(512,1024,512,51,9,16,.1,4).cuda().train()
    optimizer=initial_optim('all',1e-4,1e-6,model,'adamw');scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[150000],gamma=.05);scaler=setup_precision(model,'fp32')
    config=dict(seed=123,steps=args.steps,precision='fp32',batch_size=128,lr=1e-4,betas=[.5,.9],weight_decay=1e-6,lr_milestones=[150000],lr_gamma=.05,pkeep=.5,
        architecture=dict(embed_dim=1024,base_blocks=9,head_blocks=9,heads=16,context=51,max_motion_tokens=49,codebook=512),
        init='random',vq_sha256=prepared['vq_sha256'],data_sha256=sha256(out/'prepared.json'),clip='frozen original ViT-B/32 FP16 weights; precomputed FP32 output features',
        train_motions=len(train),validation_motions=len(val),save_every=1000,eval_every=args.eval_every,validation='custom-motion held-out takes; fixed captions; multinomial sampling T=1 seed=1230000; not official HumanML3D test',environment=environment())
    iteration=0;resume_rng=None;best={'fid':1e30,'fid_step':0,'val_ce':1e30,'val_ce_step':0}
    if args.resume:
        old=json.loads((out/'config.json').read_text())
        assert all(old[k]==v for k,v in config.items() if k not in ['environment','eval_every'])
        previous_eval_every=old['eval_every'];config=old.copy();config['eval_every']=args.eval_every
        saved=restore(args.resume,model,optimizer,scheduler,scaler,stream);iteration=saved['iteration'];best=saved['best'];resume_rng=saved['rng'];del saved
        atomic_json(out/'config.json',config)
        with (out/'resume_history.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=iteration,checkpoint=args.resume,old_eval_every=previous_eval_every,new_eval_every=args.eval_every,optimizer_scheduler_data_rng_restored=True,updated=time.time()))+'\n')
    else:atomic_json(out/'config.json',config)
    print('TRAIN_CONFIG',json.dumps(config),flush=True)
    vq=HumanVQVAE(SimpleNamespace(dataname='t2m',quantizer='ema_reset',mu=.99),512,512,512,2,2,512,3,3,'relu',None).cuda().eval()
    vq.load_state_dict(torch.load(cache/'vq_frozen.pth',map_location='cpu',weights_only=False)['net'],strict=True);vq.requires_grad_(False)
    tm=torch.tensor(np.load(cache/'meta/mean.npy'),device='cuda');ts=torch.tensor(np.load(cache/'meta/std.npy'),device='cuda')
    emean=torch.tensor(np.load(PROJECT/'checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy'),device='cuda');estd=torch.tensor(np.load(PROJECT/'checkpoints/t2m/Comp_v6_KLD005/meta/std.npy'),device='cuda')
    evaluator=EvaluatorModelWrapper(get_opt(str(PROJECT/'checkpoints/t2m/Comp_v6_KLD005/opt.txt'),torch.device('cuda')))
    writer=SummaryWriter(str(out/'tensorboard'));stopped=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *unused:stopped.__setitem__(0,True))
    def save():
        atomic_save(out/'resume_latest.pt',checkpoint(model,optimizer,scheduler,scaler,stream,iteration=iteration,best=best,config=config),rotate=True)
        atomic_save(out/'net_last.pth',dict(trans=model.state_dict(),iteration=iteration,vq_sha256=prepared['vq_sha256']))
    @torch.inference_mode()
    def validate():
        state=rng_state();seed_all(1230000);model.eval();started=time.monotonic();real=[];generated=[];r=np.zeros(3);rreal=np.zeros(3);matching=0.;match_real=0.;fallback=0;ce=0.
        atomic_json(out/'status.json',dict(stage='gpt_validation',iteration=iteration,target=args.steps,pid=os.getpid(),updated=time.time()))
        try:
            for start in range(0,len(val),32):
                inds=sorted(range(start,min(start+32,len(val))),key=lambda i:int(text['lengths'][i]),reverse=True)
                batch=[val[i] for i in inds];b=len(batch)
                cond=torch.tensor(np.stack([features[x['caption_feature_ids'][x['validation_caption_index']]] for x in batch]),device='cuda')
                target=torch.full((b,51),513,dtype=torch.long,device='cuda');lengths=[]
                gt=torch.zeros(b,196,263,device='cuda');gt_lengths=[]
                for i,row in enumerate(batch):
                    codes=row['token_variants'][0];lengths.append(len(codes));target[i,:len(codes)]=torch.tensor(codes,device='cuda');target[i,len(codes)]=512
                    motion=np.load(ROOT/'features/new_joint_vecs'/(row['name']+'.npy'))[:row['encoded_frames']];n=len(motion);gt_lengths.append(n)
                    gt[i,:n]=(torch.from_numpy(motion).cuda()-emean)/estd
                logits=model(target[:,:-1],cond)
                for i,n in enumerate(lengths):ce+=F.cross_entropy(logits[i,:n+1],target[i,:n+1]).item()
                sequences,n=sample(model,cond);fallback+=n;pred=torch.zeros_like(gt);pred_lengths=[]
                for i,codes in enumerate(sequences):
                    pose=vq.forward_decoder(codes)[0]*ts+tm;n=len(pose);pred_lengths.append(n);pred[i,:n]=(pose-emean)/estd
                word=torch.tensor(text['words'][inds],device='cuda');pos=torch.tensor(text['poses'][inds],device='cuda');lens=torch.tensor(text['lengths'][inds])
                et,er=evaluator.get_co_embeddings(word,pos,lens,gt,torch.tensor(gt_lengths));_,ep=evaluator.get_co_embeddings(word,pos,lens,pred,torch.tensor(pred_lengths))
                a,c=retrieval(et.cpu().numpy(),ep.cpu().numpy());r+=a;matching+=c
                a,c=retrieval(et.cpu().numpy(),er.cpu().numpy());rreal+=a;match_real+=c
                real.append(er.cpu().numpy());generated.append(ep.cpu().numpy())
                if (start//32)%4==0:print('VALIDATION_PROGRESS',start+b,'/',len(val),flush=True)
            real=np.concatenate(real);generated=np.concatenate(generated)
            metrics=dict(iteration=iteration,fid=fid(real,generated),top1=float(r[0]/len(val)),top2=float(r[1]/len(val)),top3=float(r[2]/len(val)),
                real_top1=float(rreal[0]/len(val)),matching=matching/len(val),real_matching=match_real/len(val),val_ce=ce/len(val),samples=len(val),
                immediate_eos_fallbacks=fallback,decoding='multinomial T=1 seed=1230000 max49',seconds=time.monotonic()-started)
            if metrics['fid']<best['fid']:
                best['fid']=metrics['fid'];best['fid_step']=iteration;atomic_save(out/'net_best_fid.pth',dict(trans=model.state_dict(),iteration=iteration,vq_sha256=prepared['vq_sha256'],metrics=metrics))
            if metrics['val_ce']<best['val_ce']:
                best['val_ce']=metrics['val_ce'];best['val_ce_step']=iteration;atomic_save(out/'net_best_val_ce.pth',dict(trans=model.state_dict(),iteration=iteration,vq_sha256=prepared['vq_sha256'],metrics=metrics))
            metrics['best']=best.copy();atomic_json(out/'validation_latest.json',metrics)
            with (out/'validation_history.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
            print('VALIDATION',json.dumps(metrics),flush=True)
            for k in ['fid','top1','top2','top3','matching','val_ce']:writer.add_scalar('Validation/'+k,metrics[k],iteration)
        finally:model.train();restore_rng(state)
    if resume_rng is not None:restore_rng(resume_rng)
    save()
    if iteration==0 or args.eval_now:validate();save()
    print('GPT_TRAINING_START',iteration,'TARGET',args.steps,flush=True);started=time.monotonic();count=0;loss_sum=0.;right_sum=0;token_sum=0
    while iteration<args.steps and not stopped[0]:
        cond,target,lengths=stream.next();cond=cond.cuda(non_blocking=True);target=target.cuda(non_blocking=True);lengths=lengths.cuda(non_blocking=True)
        inp=target[:,:-1];keep=torch.bernoulli(torch.full_like(inp,.5,dtype=torch.float32)).long();corrupted=keep*inp+(1-keep)*torch.randint_like(inp,512)
        logits=model(corrupted,cond);valid=torch.arange(51,device='cuda')[None,:]<(lengths[:,None]+1);labels=target.masked_fill(~valid,-100)
        per_token=F.cross_entropy(logits.transpose(1,2),labels,reduction='none');loss=(per_token.sum(1)/(lengths+1)).mean()
        if iteration==0:
            original=sum(F.cross_entropy(logits[i,:int(lengths[i])+1],target[i,:int(lengths[i])+1])/len(target) for i in range(len(target)))
            assert torch.allclose(loss,original,rtol=1e-6,atol=1e-6);print('ORIGINAL_LOSS_EQUIVALENCE_PASSED',flush=True)
        with torch.no_grad():sampled=torch.distributions.Categorical(logits=logits[valid]).sample();right=int((sampled==target[valid]).sum());tokens=int(valid.sum())
        norm=update(loss,model,optimizer,scaler);scheduler.step();iteration+=1;count+=1;loss_sum+=loss.item();right_sum+=right;token_sum+=tokens
        if iteration%100==0:
            seconds=(time.monotonic()-started)/count
            status=dict(stage='gpt_training',iteration=iteration,target=args.steps,pid=os.getpid(),loss=loss_sum/count,sampled_accuracy=right_sum/token_sum,
                seconds_per_step=seconds,eta_training_hours=(args.steps-iteration)*seconds/3600,lr=optimizer.param_groups[0]['lr'],best=best,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,updated=time.strftime('%Y-%m-%d %H:%M:%S'))
            atomic_json(out/'status.json',status);print('TRAIN_STATUS',json.dumps(status),flush=True)
            for k in ['loss','sampled_accuracy','lr']:writer.add_scalar('Train/'+k,status[k],iteration)
            writer.flush();started=time.monotonic();count=0;loss_sum=0.;right_sum=0;token_sum=0
        if iteration%1000==0:save();started=time.monotonic();count=0;loss_sum=0.;right_sum=0;token_sum=0
        if iteration%args.eval_every==0:validate();save();started=time.monotonic();count=0;loss_sum=0.;right_sum=0;token_sum=0
    save();writer.close()
    atomic_json(out/'status.json',dict(stage='gpt_complete' if iteration==args.steps else 'gpt_stopped',iteration=iteration,best=best,pid=os.getpid(),updated=time.time()))
    print('GPT_FINISHED_AND_SAVED',iteration,flush=True)

if __name__=='__main__':main()
