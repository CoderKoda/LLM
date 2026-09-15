"""Tiny terminal chat UI for a trained V3 model."""
from __future__ import annotations
import argparse, torch
from model import GPT,GPTConfig
from tokenizer import BPETokenizer

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='v3/checkpoints/model.pt'); p.add_argument('--tokens',type=int,default=120); p.add_argument('--temperature',type=float,default=.8); p.add_argument('--top-k',type=int,default=40); a=p.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    ckpt=torch.load(a.checkpoint,map_location=dev,weights_only=False); tok=BPETokenizer.load(ckpt['tokenizer']); model=GPT(GPTConfig(**ckpt['config'])).to(dev); model.load_state_dict(ckpt['model']); model.eval()
    history=''
    print('Koda LLM V3 | type /exit to quit')
    while True:
        try: user=input('\nYou: ').strip()
        except (EOFError,KeyboardInterrupt): break
        if user.lower()=='/exit': break
        history += f'User: {user}\nAssistant:'
        ids=torch.tensor([tok.encode(history)],dtype=torch.long,device=dev)
        out=model.generate(ids,a.tokens,a.temperature,a.top_k)[0].tolist(); text=tok.decode(out)
        answer=text[len(history):].split('\nUser:',1)[0].strip()
        print('LLM:',answer)
        history += f' {answer}\n'
if __name__=='__main__': main()
