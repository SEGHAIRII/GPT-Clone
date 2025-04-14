from dataset.dataset import get_dataloader
from model import GPT
from utilities import read_text
from config import GPT_config
import tiktoken
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()



def calc_loss_batch(input, target, model, device):
    input = input.to(device)
    target = target.to(device)
    logits = model(input)
    loss = nn.CrossEntropyLoss()
    loss = loss(logits.flatten(0,1), target.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches = None):
    loss = 0
    if len(data_loader) == 0:
        return loss
    num_batches = num_batches or len(data_loader)
    for i, (input, target) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss += calc_loss_batch(input, target, model, device)
    return loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=eval_iter
    )
    model.train()
    return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()

        print(f"Epoch {epoch + 1}/{num_epochs}")
        for input, target in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input, target, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, eval_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss.item())
                val_losses.append(eval_loss.item())
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Step {global_step}, Train Loss: {train_loss.item()}, Val Loss: {eval_loss.item()}"
                )
                
    return train_losses, val_losses, track_tokens_seen




if __name__ == "__main__":
    module_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(module_dir, "../training_corpus.txt")
    file_path = os.path.normpath(file_path)
    tokenizer = tiktoken.get_encoding("gpt2")
    input_text = read_text(file_path)
    train_ratio = 0.9
    split_idx = int(len(input_text) * train_ratio)
    train_text = input_text[:split_idx]
    val_text = input_text[split_idx:]


    train_loader = get_dataloader(train_text, tokenizer, GPT_config['context_length'], GPT_config['context_length'], batch_size=2, shuffle=True)
    val_loader = get_dataloader(
        val_text,
        tokenizer=tokenizer,
        batch_size=2,
        context_length = GPT_config["context_length"],
        stride=GPT_config["context_length"],
        shuffle=False,
        num_workers=0
    )

    
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(GPT_config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()
    
    train_losses, val_losses, track_tokens_seen = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        num_epochs= 10,
        eval_freq=5,
        eval_iter=5,
        start_context=GPT_config["context_length"],
        tokenizer=tokenizer
    )
    
    
    sentence = 'i have tried to'
    input = tokenizer.encode(sentence)
    input = torch.tensor(input).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input)
        predicted = torch.argmax(output, dim=-1)
        predicted_text = tokenizer.decode(predicted[0].cpu().numpy())
        print(predicted_text)
       
    