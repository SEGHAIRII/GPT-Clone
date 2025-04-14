import torch
import torch.nn as nn
from SubModules import LayerNorm, GelU, Linear
from Attention import MultiHeadAttention

class transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg['emb_dim'],
            d_out= cfg['emb_dim'],
            context_length= cfg['context_length'],
            num_heads= cfg['n_heads'],
            dropout= cfg['drop_rate'],
        )
        self.ff = Linear(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.dropout = nn.Dropout(cfg['drop_rate'])
        
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.norm2(x)
        x = x + shortcut
        return x
        
        