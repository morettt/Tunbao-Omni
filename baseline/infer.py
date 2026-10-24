"""Sample motion codes and decode them with the matching frozen VQ-VAE."""
import argparse
import json
import secrets
import sys
from pathlib import Path

from paths import DATA_ROOT, T2M_ROOT, CLIP_PATH
sys.path.insert(0, str(T2M_ROOT))
import clip
import numpy as np
import torch
from types import SimpleNamespace
from models.t2m_trans import Text2Motion_Transformer
from models.vqvae import HumanVQVAE
from utils.motion_process import recover_from_ric
from cached_gpt_eval import CachedGPT
from common import sha256

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', required=True, help='English motion description')
    parser.add_argument('--checkpoint', type=Path, default=DATA_ROOT/'gpt/net_best_fid.pth')
    parser.add_argument('--vq', type=Path, default=DATA_ROOT/'tokens/vq_frozen.pth')
    parser.add_argument('--norm-dir', type=Path, default=DATA_ROOT/'tokens/meta')
    parser.add_argument('--out', type=Path, default=Path('outputs/motion'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--greedy', action='store_true', help='Opt in to greedy decoding; default is random sampling')
    args = parser.parse_args()
    if args.temperature <= 0:
        parser.error('--temperature must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('The current cached decoder requires a CUDA GPU')
    if not CLIP_PATH.is_file():
        raise FileNotFoundError('Set CLIP_MODEL_PATH to the OpenAI CLIP ViT-B/32 weight file')
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    torch.manual_seed(seed); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['vq_sha256'] != sha256(args.vq):
        raise ValueError('GPT and VQ checkpoints do not match')
    model = Text2Motion_Transformer(512,1024,512,51,9,16,.1,4).cuda().eval()
    model.load_state_dict(checkpoint['trans'], strict=True)
    vq = HumanVQVAE(SimpleNamespace(dataname='t2m',quantizer='ema_reset',mu=.99),512,512,512,2,2,512,3,3,'relu',None).cuda().eval()
    vq.load_state_dict(torch.load(args.vq,map_location='cpu',weights_only=False)['net'],strict=True)
    encoder,_ = clip.load(str(CLIP_PATH),device='cuda',jit=False)
    encoder.eval(); codes=[]; last=None
    with torch.inference_mode():
        condition=encoder.encode_text(clip.tokenize([args.text],truncate=True).cuda()).float()
        cache=CachedGPT(model,max_batch=1)
        for position in range(50):
            logits=cache.step(last,condition,position)
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite generation logits')
            code=logits.argmax(-1) if args.greedy else torch.distributions.Categorical(logits=logits/args.temperature).sample()
            value=int(code.item())
            if value==512 or len(codes)==49:
                break
            codes.append(value);last=code
        if not codes:
            raise RuntimeError('The model ended immediately; try a different seed or description')
        mean=torch.tensor(np.load(args.norm_dir/'mean.npy'),device='cuda')
        std=torch.tensor(np.load(args.norm_dir/'std.npy'),device='cuda')
        features=vq.forward_decoder(torch.tensor([codes],device='cuda'))*std+mean
        joints=recover_from_ric(features.float(),22)[0].cpu().numpy()
    assert joints.shape==(len(codes)*4,22,3) and np.isfinite(joints).all()
    args.out.mkdir(parents=True,exist_ok=True)
    np.save(args.out/'codes.npy',np.asarray(codes,dtype=np.int64))
    np.save(args.out/'joints.npy',joints)
    metadata=dict(text=args.text,seed=seed,fps=20,frames=len(joints),tokens=len(codes),
                  checkpoint_step=checkpoint['iteration'],decoding='greedy' if args.greedy else 'multinomial',temperature=args.temperature)
    (args.out/'motion.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False))

if __name__ == '__main__':
    main()
