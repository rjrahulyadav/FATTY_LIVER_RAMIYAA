import torch
from pathlib import Path
p=Path('best_model.pth')
state=torch.load(p,map_location='cpu')
print('type',type(state))
if isinstance(state,dict):
    ks=list(state.keys())
    print('keys sample:', ks[:30])
    for k in ks[:50]:
        v=state[k]
        print('\nkey:',k,'type',type(v))
        if hasattr(v,'dtype'):
            print(' shape',getattr(v,'shape',None),'dtype',v.dtype,'nan?',torch.isnan(v).any().item())
            break
else:
    print('not a dict; object repr length',len(repr(state)))
