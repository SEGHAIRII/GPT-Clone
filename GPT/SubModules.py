import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
    
class GelU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(2 / torch.pi) * (x + 0.044715 * x ** 3)))
    
    
class Linear(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GelU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    