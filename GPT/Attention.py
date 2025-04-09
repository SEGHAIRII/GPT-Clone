import torch.nn as nn
import torch


class Attention(nn.Module):
    def __init__(self, d_in:int, d_out:int):
        super(Attention, self).__init__()
        self.W_keys = nn.Linear(d_in, d_out, bias=False)
        self.W_queries = nn.Linear(d_in, d_out, bias=False)
        self.W_values = nn.Linear(d_in, d_out, bias=False)
        
    def forward(self, X):
        keys = self.W_keys(X)
        queries = self.W_queries(X)
        values = self.W_values(X)
        
        scores = torch.matmul(queries, keys.T) / (keys.size(-1) ** 0.5)
        
        attn_weights = nn.functional.softmax(scores, dim=-1)
        
        context_vect = torch.matmul(attn_weights, values)
        
        return context_vect
    
    
    