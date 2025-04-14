from dataset.dataset import get_dataloader
from utilities import read_text
from config import GPT_config
import tiktoken
import torch
import torch.nn as nn

tokenizer = tiktoken.get_encoding("gpt2")
input_text = read_text("../training_corpus.txt")
train_ratio = 0.9
split_idx = int(len(input_text) * train_ratio)
train_text = input_text[:split_idx]
val_text = input_text[split_idx:]


train_loader = get_dataloader(train_text, tokenizer, GPT_config['context_size'], GPT_config['context_size'], batch_size=2, shuffle=True)
val_loader = get_dataloader(
    val_text,
    batch_size=2,
    max_length = GPT_config["context_length"],
    stride=GPT_config["context_length"],
    shuffle=False,
    num_workers=0
)
