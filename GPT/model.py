import torch
import torch.nn as nn
from Transformer import transformer
from SubModules import LayerNorm


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop = nn.Dropout(cfg['drop_rate'])
        self.transf_blocks = nn.Sequential(
            *[
                transformer(cfg) for _ in range(cfg['n_layers'])
            ]
        )    
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
        
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.size()
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop(x)
        x = self.transf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
        
        