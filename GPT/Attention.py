import torch.nn as nn
import torch


class MaskedAttention(nn.Module):
    def __init__(self, d_in:int, d_out:int, context_length:int, dropout:float=0.0):
        super().__init__()
        self.d_out = d_out
        self.W_keys = nn.Linear(d_in, d_out, bias=False)
        self.W_queries = nn.Linear(d_in, d_out, bias=False)
        self.W_values = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal=-1)
        )
        
        
    def forward(self, X):
        b, num_tokens, d_in = X.size()
        keys = self.W_keys(X)
        queries = self.W_queries(X)
        values = self.W_values(X)
        
        scores = torch.matmul(queries, keys.transpose(1,2))
        
        scores.masked_fill_(self.mask[:num_tokens, :num_tokens] == 0, - torch.inf)
        attn_weights = torch.softmax(scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vect = torch.matmul(attn_weights, values)
        
        return context_vect
    
    
    
    