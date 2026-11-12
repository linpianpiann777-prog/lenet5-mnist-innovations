"""Export actual per-layer shapes, trainable parameters and Conv/Linear MACs."""
import json
import torch
from torch import nn
from data_utils import ROOT
from models import LeNet5, VARIANTS, complexity

def main():
    profiles={}
    for v in VARIANTS:
        model=LeNet5(v).eval();rows=[];handles=[]
        for name,m in model.named_modules():
            if isinstance(m,(nn.Conv2d,nn.Linear,nn.AvgPool2d)):
                def hook(layer,inputs,out,name=name):
                    mac=(out.numel()*(layer.in_channels//layer.groups)*layer.kernel_size[0]*layer.kernel_size[1] if isinstance(layer,nn.Conv2d)
                         else out.numel()*layer.in_features if isinstance(layer,nn.Linear) else 0)
                    rows.append(dict(name=name,type=type(layer).__name__,input=list(inputs[0].shape),output=list(out.shape),
                                     parameters=sum(p.numel() for p in layer.parameters()),macs=mac))
                handles.append(m.register_forward_hook(hook))
        with torch.no_grad():model(torch.zeros(1,1,32,32))
        for h in handles:h.remove()
        profiles[v]={'layers':rows,**complexity(model)}
        assert sum(r['parameters'] for r in rows)==profiles[v]['parameters']
        assert sum(r['macs'] for r in rows)==profiles[v]['macs']
    (ROOT/'results/model_profiles.json').write_text(json.dumps(profiles,indent=2))
    for v,r in profiles.items():print(v,r['parameters'],r['macs'])

if __name__=='__main__':main()
