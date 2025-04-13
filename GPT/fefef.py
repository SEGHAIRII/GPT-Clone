import torch

from Transformer import transformer
from config import GPT_config

torch.manual_seed(123)
x = torch.rand(2, 4, 768)
print('input shape : ', x.shape)

block = transformer(GPT_config)
output = block(x)
print('output shape : ', output.shape)
