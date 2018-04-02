import torch
import torch.nn as nn

class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_groups,1))
        self.bias = nn.Parameter(torch.zeros(num_groups,1))
        self.num_groups = num_groups
        self.eps = eps
        self.mean = None
        self.var = None

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        x = (x - x.mean(-1, keepdim=True)) /  (x.var(-1, keepdim=True) + self.eps).sqrt()

        #mean = x.mean(-1, keepdim=True)
        #var = x.var(-1, keepdim=True)
        #x = (x-mean) / (var+self.eps).sqrt()
        x = x * self.weight + self.bias

        return x.view(N,C,H,W)
