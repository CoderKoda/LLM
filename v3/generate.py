"""Generate text from a V3 checkpoint."""
from __future__ import annotations
import argparse, torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer

def device(name):
    if name=='cpu': return torch.device('cpu')
    if name=='cuda': return torch.device('cuda') if torch.cuda.is_available() else (_ for _ in ()).throw(RuntimeError('CUDA unavailable'))
    if name=='mps': return torch.device('mps') if torch.backends.mps.is_available() else (_ for _ in ()).throw(RuntimeError('MPS unavailable'))
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='v3/checkpoints/model.pt'); p.add_argument('--prompt',default=''); p.add_argument('--tokens',type=int,default=200); p.add_argument('--temperature',type=float,default=.8); p.add_argument('--top-k',type=int,default=40); p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto'); a=p.parse_args()
    dev=device(a.device); ckpt=torch.load(a.checkpoint,map_location=dev,weights_only=False); tok=BPETokenizer.load(ckpt['tokenizer']); model=GPT(GPTConfig(**ckpt['config'])).to(dev); model.load_state_dict(ckpt['model']); ids=torch.tensor([tok.encode(a.prompt)],dtype=torch.long,device=dev)
    if ids.shape[1]==0: ids=torch.tensor([[32]],dtype=torch.long,device=dev)
    print(tok.decode(model.generate(ids,a.tokens,a.temperature,a.top_k)[0].tolist()))
if __name__=='__main__': main()
