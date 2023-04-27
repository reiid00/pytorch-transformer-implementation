import torch
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def generate_masks(src, tgt, pad_idx):

    # masking for the src
    # Used to prevent the model (encoder) from attending to padding tokens
    # unsqueeze function to add two dimensions, required for the subsequent
    # attention mechanism's broadcasting
    # shape : (batch_size, 1, 1, src_len)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # masking for the target
    # Used to prevent the model (decoder) from attending to padding tokens and future tokens
    # unsqueeze function to add two dimensions, required for the subsequent
    # attention mechanism's broadcasting
    # shape : (batch_size, 1, tgt_len, 1)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)

    tgt_len = tgt.size(1)

   
    # Creating the upper triangle to prevend the decoder to visualize future tokens in the target sequnce

    # multiple lines

    # Create a square matrix filled with ones, of shape (tgt_len, tgt_len)
    # ones_matrix = torch.ones(tgt_len, tgt_len, device=tgt.device)
    # Create an upper triangular matrix filled with ones and zeros
    # upper_triangular = torch.triu(ones_matrix)
    # Convert the upper triangular matrix to a boolean tensor
    # bool_upper_triangular = upper_triangular.bool()
    # Add two dimensions to the boolean tensor, resulting in a shape of (1, 1, tgt_len, tgt_len)
    # bool_upper_triangular_expanded = bool_upper_triangular.unsqueeze(0).unsqueeze(0)
    # Perform a logical AND operation between tgt_mask and the expanded boolean tensor
    # tgt_mask = tgt_mask & bool_upper_triangular_expanded

    # In 1 line
    tgt_mask = tgt_mask & torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool().unsqueeze(0).unsqueeze(0)

    return src_mask, tgt_mask

def batch_to_tensor(batch, pad_idx, device):
    # Max length of sequences in the batch
    max_len = max([len(x) for x in batch])

    # create a tensor of shape (batch_size, max_len) filled with the padding index
    tensor = torch.full((len(batch), max_len), pad_idx, dtype=torch.long, device=device)

    # Iterate through the batch and replace the corresponding row with the sequence's tokens
    for i, seq in enumerate(batch):
        tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    return tensor


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
    }, path)

def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return epoch


def get_training_state(training_config):
    return {
        'epoch': training_config['epoch'],
        'train_loss': training_config['train_loss'],
        'val_loss': training_config['val_loss'],
        'bleu_score': training_config['bleu_score'],
    }

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Simple bleu calculation, may improve it later
def calculate_bleu(reference, hypothesis):
    smooth = SmoothingFunction().method4
    return sentence_bleu([reference], hypothesis, smoothing_function=smooth)

def evaluate_model(model, data_loader, criterion, device, pad_idx):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask = generate_masks(src, tgt, pad_idx)

            output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1, :-1])
            loss = criterion(output, tgt[:, 1:])
            total_loss += loss.item() * (tgt[:, 1:] != pad_idx).sum().item()
            total_tokens += (tgt[:, 1:] != pad_idx).sum().item()

    model.train()
    return total_loss / total_tokens

def evaluate_metrics(model, data_loader, tgt_vocab, pad_idx, device):
    model.eval()
    bleu_scores = []
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask = generate_masks(src, pad_idx)
            output = model.generate(src, src_mask)
            hypothesis = [tgt_vocab.itos[token] for token in output if token not in (pad_idx, tgt_vocab.stoi['<sos>'], tgt_vocab.stoi['<eos>'])]
            reference = [tgt_vocab.itos[token] for token in tgt if token not in (pad_idx, tgt_vocab.stoi['<sos>'], tgt_vocab.stoi['<eos>'])]
            bleu = calculate_bleu(reference, hypothesis)
            bleu_scores.append(bleu)

    model.train()
    return sum(bleu_scores) / len(bleu_scores)


