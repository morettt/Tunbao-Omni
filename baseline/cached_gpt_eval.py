from paths import DATA_ROOT, T2M_ROOT, CLIP_PATH
"""Incremental FP32 inference, verified against the unmodified upstream GPT."""
import math
import torch
from torch.nn import functional as F

class CachedGPT:
    def __init__(self,model,max_batch=32):
        self.model=model;self.blocks=list(model.trans_base.blocks)+list(model.trans_head.blocks)
        h=self.blocks[0].attn.n_head;d=model.trans_base.tok_emb.embedding_dim//h
        self.keys=[torch.empty(max_batch,h,model.block_size,d,device='cuda') for _ in self.blocks]
        self.values=[torch.empty_like(k) for k in self.keys]
    @torch.inference_mode()
    def step(self,last,features,position):
        b=len(features);base=self.model.trans_base
        x=base.cond_emb(features).unsqueeze(1) if position==0 else base.tok_emb(last[:,None])
        x=x+base.pos_embed.embed[position:position+1].transpose(0,1)
        for i,block in enumerate(self.blocks):
            a=block.attn;n=block.ln1(x);h=a.n_head;d=x.shape[-1]//h
            q=a.query(n).view(b,1,h,d).transpose(1,2)
            k=a.key(n).view(b,1,h,d).transpose(1,2);v=a.value(n).view(b,1,h,d).transpose(1,2)
            self.keys[i][:b,:,position:position+1]=k;self.values[i][:b,:,position:position+1]=v
            weights=F.softmax((q@self.keys[i][:b,:,:position+1].transpose(-2,-1))*(1.0/math.sqrt(d)),dim=-1)
            y=(weights@self.values[i][:b,:,:position+1]).transpose(1,2).contiguous().view(b,1,h*d)
            x=x+a.proj(y);x=x+block.mlp(block.ln2(x))
        return self.model.trans_head.head(self.model.trans_head.ln_f(x))[:,-1]
    @torch.inference_mode()
    def sample(self,features,seed=None,stochastic=True):
        if seed is not None:torch.cuda.manual_seed(seed)
        b=len(features);codes=torch.empty((b,0),dtype=torch.long,device='cuda');lengths=torch.full((b,),-1,dtype=torch.long,device='cuda')
        last=None
        for k in range(self.model.block_size):
            logits=self.step(last,features,k)
            assert torch.isfinite(logits).all(),'Nonfinite cached logits'
            nxt=torch.distributions.Categorical(probs=F.softmax(logits,dim=-1)).sample() if stochastic else logits.argmax(-1)
            lengths[(nxt==self.model.num_vq)&(lengths<0)]=k
            codes=torch.cat((codes,nxt[:,None]),1);last=nxt
            if (lengths>=0).all():break
        lengths[lengths<0]=self.model.block_size-1
        fallback=int((lengths==0).sum())
        sequences=[torch.ones((1,1),dtype=torch.long,device='cuda') if int(n)==0 else codes[i:i+1,:int(n)] for i,n in enumerate(lengths)]
        return sequences,fallback
    @torch.inference_mode()
    def verify(self,features):
        features=features[:4];prefix=torch.empty((len(features),0),dtype=torch.long,device='cuda');last=None;max_error=0.
        for k in range(self.model.block_size):
            cached=self.step(last,features,k);original=self.model([] if k==0 else prefix,features)[:,-1]
            error=float((cached-original).abs().max());max_error=max(max_error,error)
            assert torch.allclose(cached,original,atol=1e-4,rtol=1e-4),(k,error)
            state=torch.cuda.get_rng_state();a=torch.distributions.Categorical(probs=F.softmax(cached,-1)).sample()
            torch.cuda.set_rng_state(state);b=torch.distributions.Categorical(probs=F.softmax(original,-1)).sample()
            assert torch.equal(a,b),('Coupled categorical mismatch',k)
            prefix=torch.cat((prefix,a[:,None]),1);last=a
        for i in range(4):
            for stochastic in [False,True]:
                torch.cuda.manual_seed(20261021+i)
                try:expected=self.model.sample(features[i:i+1],stochastic)
                except UnboundLocalError:expected=torch.ones((1,1),device='cuda',dtype=torch.long)
                actual,_=self.sample(features[i:i+1],seed=20261021+i,stochastic=stochastic)
                assert torch.equal(expected,actual[0]),('Original sampler differs',i,stochastic)
        return dict(passed=True,positions_verified=51,batch_size=4,max_logit_error=max_error,original_sampler_cases=8,
                    coupled_stochastic_tokens_identical=True)
