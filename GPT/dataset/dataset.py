import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class gptDataSet(Dataset):
    def __init__(self, text:str, tokenizer:tiktoken.core.Encoding, context_size: int, stride:int):
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.stride = stride
        self.inputs = []
        self.outputs = []
        tokens = tokenizer.encode(text)
        for i in range(0, len(tokens) - context_size, stride):
            input_tokens = tokens[i:i + context_size ]
            output_tokens = tokens[i + 1:i + context_size + 1]
            self.inputs.append(torch.as_tensor(input_tokens))
            self.outputs.append(torch.as_tensor(output_tokens))
            
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    
    
    
def get_dataloader(text:str, tokenizer:tiktoken.core.Encoding, context_length:int, stride:int, batch_size:int, shuffle:bool=True, num_workers:int=0):
    dataset = gptDataSet(text, tokenizer, context_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

