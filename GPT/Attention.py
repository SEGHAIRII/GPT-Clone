import torch.nn as nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in:int, d_out:int, context_length:int, dropout:float=0.0, num_heads:int=1, qkv_bias:bool = False):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        # Here we used one big matrix to store all matrices
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.outproj = nn.Linear(d_out, d_out)
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal=-1)
        )
        
        
    def forward(self, X):
        b, num_tokens, d_in = X.size()
        keys = self.W_key(X)
        queries = self.W_query(X)
        values = self.W_value(X)
        # Here we split the big matrix into num_heads matrices
        # and transpose the last two dimensions
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)
        
        
        scores = (queries @ keys.transpose(2,3)) / self.head_dim ** 0.5
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=X.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1,2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.outproj(context_vec)
        return context_vec
    
    
    
    