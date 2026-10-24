from paths import DATA_ROOT, T2M_ROOT, CLIP_PATH
"""Official T2M-GPT VQ model on custom-motion, with held-out reconstruction validation."""
import os,sys,time,json,signal,argparse,random,fcntl
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MPLBACKEND','Agg')
BASE=DATA_ROOT;PROJECT=T2M_ROOT
sys.path[:0]=[str(PROJECT)];os.chdir(PROJECT)
import numpy as np
import torch
from types import SimpleNamespace
from scipy.linalg import sqrtm
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from models.vqvae import HumanVQVAE
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.motion_process import recover_from_ric
from utils.losses import ReConsLoss
from common import seed_all,BatchStream,checkpoint,restore,atomic_save,atomic_json,autocast,setup_precision,update,environment,sha256,rng_state,restore_rng

class Motions(Dataset):
    def __init__(self,data,ids,mean,std):
        self.mean,self.std=mean,std;self.ids=ids
        self.data=[((np.load(data/'new_joint_vecs'/(key+'.npy'))-mean)/std).astype(np.float32) for key in ids]
        assert all(len(x)>=64 and np.isfinite(x).all() for x in self.data)
    def __len__(self):return len(self.data)
    def __getitem__(self,index):
        x=self.data[index];start=random.randint(0,len(x)-64);return x[start:start+64]

def fid(real,pred):
    a,b=np.mean(real,axis=0),np.mean(pred,axis=0);c,d=np.cov(real,rowvar=False),np.cov(pred,rowvar=False)
    cross=sqrtm(c@d)
    if not np.isfinite(cross).all():cross=sqrtm((c+np.eye(len(c))*1e-6)@(d+np.eye(len(d))*1e-6))
    if np.iscomplexobj(cross):
        assert np.max(np.abs(cross.imag))<.01,'Unstable FID square root'
        cross=cross.real
    value=float((a-b)@(a-b)+np.trace(c+d-2*cross));assert np.isfinite(value)
    return max(value,0.)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out-dir',default=str(BASE/'vq'))
    parser.add_argument('--steps',type=int,default=300000);parser.add_argument('--resume');parser.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    args=parser.parse_args();out=Path(args.out_dir);out.mkdir(exist_ok=True)
    lock=(out/'.training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'config.json').exists() and not args.resume:raise RuntimeError('Use explicit resume for existing run')
    seed_all(123);data=BASE/'features';mean=np.load(out/'meta/mean.npy');std=np.load(out/'meta/std.npy')
    train_ids=(out/'train.txt').read_text().splitlines();val_ids=(out/'val.txt').read_text().splitlines()
    ds=Motions(data,train_ids,mean,std);val=Motions(data,val_ids,mean,std)
    config=dict(seed=123,steps=args.steps,warmup=1000,batch_size=256,window_size=64,lr=2e-4,lr_milestones=[200000],lr_gamma=.05,
        precision=args.precision,architecture='official T2M-GPT HumanVQVAE, codebook512 dim512 width512 depth3 down_t2 stride2 dilation3',
        initialization='random',quantizer='ema_reset',mu=.99,commit_weight=.02,position_weight=.5,loss='l1_smooth',
        eval_every=1000,save_every=1000,log_every=100,validation='custom-motion held-out takes: fixed windows, reconstruction FID and MPJPE; not official HumanML3D test scores',
        dataset_sha256=sha256(out/'dataset.json'),normalization_sha256=[sha256(out/'meta/mean.npy'),sha256(out/'meta/std.npy')],environment=environment())
    if args.resume:
        old=json.loads((out/'config.json').read_text())
        for k in ['steps','precision','dataset_sha256','normalization_sha256']:assert old[k]==config[k],k
    else:atomic_json(out/'config.json',config)
    print('TRAIN_CONFIG',json.dumps(config),flush=True)
    stream=BatchStream(ds,256,123,workers=4)
    model=HumanVQVAE(SimpleNamespace(dataname='t2m',quantizer='ema_reset',mu=.99),512,512,512,2,2,512,3,3,'relu',None).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[200000],gamma=.05)
    scaler=setup_precision(model,args.precision);loss_fn=ReConsLoss('l1_smooth',22);writer=SummaryWriter(str(out/'tensorboard'))
    iteration=warmup=0;best={'fid':1e30,'mpjpe_mm':1e30,'fid_step':0,'mpjpe_step':0}
    if args.resume:
        saved=restore(args.resume,model,optimizer,scheduler,scaler,stream);iteration=saved['iteration'];warmup=saved['warmup_done'];best=saved['best'];del saved
    evaluator=EvaluatorModelWrapper(get_opt(str(PROJECT/'checkpoints/t2m/Comp_v6_KLD005/opt.txt'),torch.device('cuda')))
    eval_mean=torch.tensor(np.load(PROJECT/'checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy'),device='cuda')
    eval_std=torch.tensor(np.load(PROJECT/'checkpoints/t2m/Comp_v6_KLD005/meta/std.npy'),device='cuda')
    tm=torch.tensor(mean,device='cuda');ts=torch.tensor(std,device='cuda')
    # Each held-out sequence supplies at most two fixed, nonrandom windows.
    windows=[]
    for x in val.data:
        length=min(196,len(x)//4*4)
        for start in sorted({0,len(x)-length}):windows.append(torch.from_numpy(x[start:start+length]))
    atomic_json(out/'validation_protocol.json',dict(take_disjoint=True,clips=len(val),windows=len(windows),max_frames=196,fps=20,
        evaluator='HumanML3D pretrained motion embedding, official evaluator normalization',reported_as='custom-motion reconstruction FID, not comparable to HumanML3D benchmark FID'))
    stopped=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *unused:stopped.__setitem__(0,True))
    def save():
        atomic_save(out/'resume_latest.pt',checkpoint(model,optimizer,scheduler,scaler,stream,iteration=iteration,warmup_done=warmup,best=best,config=config),rotate=True)
        atomic_save(out/'net_last.pth',{'net':model.state_dict(),'iteration':iteration,'warmup_done':warmup})
    @torch.no_grad()
    def evaluate():
        state=rng_state();model.eval();real=[];pred=[];error_sum=0.;joint_count=0;feature_sum=0.;feature_count=0;started=time.monotonic()
        try:
            for offset in range(0,len(windows),32):
                batch=windows[offset:offset+32];lengths=torch.tensor([len(x) for x in batch],device='cuda')
                x=torch.zeros(len(batch),196,263,device='cuda')
                for i,item in enumerate(batch):x[i,:len(item)]=item.cuda()
                y=torch.zeros_like(x)
                for length in lengths.unique().tolist():
                    indices=torch.where(lengths==length)[0]
                    with autocast(args.precision):output,_,_=model(x[indices,:length])
                    y[indices,:length]=output.float()
                source=x.float()*ts+tm;reconstruction=y.float()*ts+tm
                # Padding must stay zero in evaluator space after normalization.
                a=torch.zeros_like(x);b=torch.zeros_like(x)
                for i,length in enumerate(lengths.tolist()):
                    a[i,:length]=(source[i,:length]-eval_mean)/eval_std;b[i,:length]=(reconstruction[i,:length]-eval_mean)/eval_std
                    gt=recover_from_ric(source[i,:length],22);pr=recover_from_ric(reconstruction[i,:length],22)
                    error_sum+=torch.linalg.vector_norm(gt-pr,dim=-1).sum().item();joint_count+=length*22
                    feature_sum+=torch.abs(y[i,:length].float()-x[i,:length]).sum().item();feature_count+=length*263
                real.append(evaluator.get_motion_embeddings(a,lengths).cpu().numpy());pred.append(evaluator.get_motion_embeddings(b,lengths).cpu().numpy())
            metrics=dict(iteration=iteration,warmup_done=warmup,bones_reconstruction_fid=fid(np.concatenate(real),np.concatenate(pred)),mpjpe_mm=error_sum/joint_count*1000,
                normalized_feature_l1=feature_sum/feature_count,windows=len(windows),seconds=time.monotonic()-started)
            if metrics['bones_reconstruction_fid']<best['fid']:
                best['fid']=metrics['bones_reconstruction_fid'];best['fid_step']=iteration
                atomic_save(out/'net_best_fid.pth',{'net':model.state_dict(),'iteration':iteration,'metrics':metrics})
            if metrics['mpjpe_mm']<best['mpjpe_mm']:
                best['mpjpe_mm']=metrics['mpjpe_mm'];best['mpjpe_step']=iteration
                atomic_save(out/'net_best_mpjpe.pth',{'net':model.state_dict(),'iteration':iteration,'metrics':metrics})
            metrics['best']=best.copy();atomic_json(out/'validation_latest.json',metrics)
            with (out/'validation_history.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
            for k in ['bones_reconstruction_fid','mpjpe_mm','normalized_feature_l1']:writer.add_scalar('Validation/'+k,metrics[k],iteration)
            print('VALIDATION',json.dumps(metrics),flush=True)
        finally:model.train();restore_rng(state)
    def step():
        x=stream.next().cuda(non_blocking=True).float()
        with autocast(args.precision):
            y,commit,ppl=model(x);recons=loss_fn(y,x);position=loss_fn.forward_vel(y,x);loss=recons+.5*position+.02*commit
        norm=update(loss,model,optimizer,scaler)
        return dict(loss=float(loss.detach()),recons=float(recons.detach()),position=float(position.detach()),commit=float(commit.detach()),perplexity=float(ppl.detach()),grad_norm=norm)
    def status(stage,metrics,seconds):
        value=dict(stage=stage,pid=os.getpid(),iteration=iteration,warmup_done=warmup,target=args.steps,seconds_per_step=seconds,
            eta_training_hours=max(0,args.steps-iteration+999-warmup)*seconds/3600,best=best,lr=optimizer.param_groups[0]['lr'],
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,updated=time.strftime('%Y-%m-%d %H:%M:%S'),**metrics)
        atomic_json(out/'status.json',value);print('TRAIN_STATUS',json.dumps(value),flush=True)
        for k,v in metrics.items():writer.add_scalar('Train/'+k,v,iteration+warmup)
        writer.flush()
    print('WARMUP_START',warmup,'TARGET',999,flush=True);start=time.monotonic();count=0;sums={}
    while warmup<999 and not stopped[0]:
        for group in optimizer.param_groups:group['lr']=2e-4*(warmup+2)/1001
        metrics=step();warmup+=1;count+=1
        for k,v in metrics.items():sums[k]=sums.get(k,0)+v
        if warmup%100==0:status('warmup',{k:v/count for k,v in sums.items()},(time.monotonic()-start)/count);start=time.monotonic();count=0;sums={}
        if warmup%500==0:save()
    save()
    if stopped[0]:print('STOPPED_AND_SAVED',flush=True);return
    if iteration==0:evaluate();save()
    print('MAIN_TRAINING_START',iteration,'TARGET',args.steps,flush=True);start=time.monotonic();count=0;sums={}
    while iteration<args.steps and not stopped[0]:
        metrics=step();scheduler.step();iteration+=1;count+=1
        for k,v in metrics.items():sums[k]=sums.get(k,0)+v
        if iteration%100==0:status('vq_training',{k:v/count for k,v in sums.items()},(time.monotonic()-start)/count);start=time.monotonic();count=0;sums={}
        if iteration%1000==0:evaluate();save();start=time.monotonic();count=0;sums={}
    save();writer.close()
    if iteration==args.steps:atomic_json(out/'complete.json',dict(iteration=iteration,best=best,finished=time.time()));print('VQ_TRAINING_COMPLETE',flush=True)
    else:print('STOPPED_AND_SAVED',iteration,flush=True)

if __name__=='__main__':main()
