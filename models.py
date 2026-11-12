"""LeNet-5 adaptation: dense C3, ordinary average pooling, softmax head.

Unlike the 1998 model, C3 is fully connected across channels, subsampling has
no trainable scale/bias, and cross entropy replaces the Euclidean RBF head.
C3-only is a localization ablation. DW factorizes both C3 and C5 to address
the compute hotspot (C3) and the parameter hotspot (C5) separately.
"""
import torch
from torch import nn

VARIANTS = ('baseline', 'dw_c3', 'dw', 'se', 'dw_se')


class SE(nn.Module):
    def __init__(self, channels=16, reduction=4):
        super().__init__()
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(channels, channels // reduction, 1),
                                  nn.ReLU(), nn.Conv2d(channels // reduction, channels, 1),
                                  nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


class LeNet5(nn.Module):
    def __init__(self, variant='baseline'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.c1 = nn.Conv2d(1, 6, 5)
        self.c3 = (nn.Sequential(nn.Conv2d(6, 6, 5, groups=6, bias=False),
                                 nn.Conv2d(6, 16, 1))
                   if variant in ('dw_c3', 'dw', 'dw_se') else nn.Conv2d(6, 16, 5))
        self.se = SE() if variant in ('se', 'dw_se') else nn.Identity()
        self.pool = nn.AvgPool2d(2)
        self.c5 = (nn.Sequential(nn.Conv2d(16, 16, 5, groups=16, bias=False),
                                 nn.Conv2d(16, 120, 1))
                   if variant in ('dw', 'dw_se') else nn.Conv2d(16, 120, 5))
        self.f6 = nn.Linear(120, 84)
        self.head = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.tanh(self.c1(x)))
        x = self.pool(self.se(torch.tanh(self.c3(x))))
        x = torch.tanh(self.c5(x)).flatten(1)
        return self.head(torch.tanh(self.f6(x)))


def complexity(model):
    """Conv/Linear MACs at batch=1, 32x32. FLOPs=2*MACs.
    Excludes bias, nonlinearities, pooling and elementwise gating.
    """
    macs, handles = [], []
    def hook(m, inputs, out):
        if isinstance(m, nn.Conv2d):
            macs.append(out.numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1])
        else:
            macs.append(out.numel() * m.in_features)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(hook))
    mode = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 32, 32, device=next(model.parameters()).device))
    for h in handles:
        h.remove()
    model.train(mode)
    return dict(parameters=sum(p.numel() for p in model.parameters()), macs=sum(macs), flops=2*sum(macs))
