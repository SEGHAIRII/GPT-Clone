import torch.nn as nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in:int, d_out:int, context_length:int, dropout:float=0.0, num_heads:int=1):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        # Here we used one big matrix to store all matrices
        self.W_keys = nn.Linear(d_in, d_out, bias=False)
        self.W_queries = nn.Linear(d_in, d_out, bias=False)
        self.W_values = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.outproj = nn.Linear(d_out, d_out)
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal=-1)
        )
        
        
    def forward(self, X):
        b, num_tokens, d_in = X.size()
        keys = self.W_keys(X)
        queries = self.W_queries(X)
        values = self.W_values(X)
        # Here we split the big matrix into num_heads matrices
        # and transpose the last two dimensions
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        
        scores = torch.matmul(queries, keys.transpose(2,3))
        
        scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], - torch.inf)
        attn_weights = torch.softmax(scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = torch.matmul(attn_weights, values).transpose(1,2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.outproj(context_vec)
        
        
        return context_vec
    
    
    
    