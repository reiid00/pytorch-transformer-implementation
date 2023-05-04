import torch
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

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
    
    tgt_len = tgt.size(1)
    tgt_padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
   
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
    tgt_look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool().unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_padding_mask & tgt_look_ahead_mask.expand(tgt_padding_mask.size(0), -1, -1, -1)

    return src_mask, tgt_mask

def generate_masks_new(src, tgt, src_pad_idx, tgt_pad_idx):

    # masking for the src
    # Used to prevent the model (encoder) from attending to padding tokens
    # unsqueeze function to add two dimensions, required for the subsequent
    # attention mechanism's broadcasting
    # shape : (batch_size, 1, 1, src_len)
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

    # masking for the target
    # Used to prevent the model (decoder) from attending to padding tokens and future tokens
    # unsqueeze function to add two dimensions, required for the subsequent
    # attention mechanism's broadcasting
    # shape : (batch_size, 1, tgt_len, 1)
    
    tgt_len = tgt.size(1)
    tgt_padding_mask = (tgt != tgt_pad_idx).unsqueeze(1).unsqueeze(3)
   
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
    tgt_look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool().unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_padding_mask & tgt_look_ahead_mask.expand(tgt_padding_mask.size(0), -1, -1, -1)

    return src_mask, tgt_mask

def generate_src_mask(src, src_pad_idx):
    return (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

def generate_tgt_mask(tgt, tgt_pad_idx):
    tgt_len = tgt.size(1)
    tgt_padding_mask = (tgt != tgt_pad_idx).unsqueeze(1).unsqueeze(3)
    tgt_look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool().unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_padding_mask & tgt_look_ahead_mask.expand(tgt_padding_mask.size(0), -1, -1, -1)

    return tgt_mask

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
    return model, optimizer, scheduler, epoch


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


def evaluate_model(model, data_loader, criterion, device, src_pad_idx, tgt_pad_idx):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask = generate_masks_new(src, tgt, src_pad_idx, tgt_pad_idx)

            output = model(src, tgt, src_mask, tgt_mask)
            _, loss = criterion(output, tgt)
            total_loss += loss * (tgt != tgt_pad_idx).sum().item()
            total_tokens += (tgt != tgt_pad_idx).sum().item()

    model.train()
    return total_loss / total_tokens

def batch_greedy_decode_v2(model, src, src_mask, max_len, tgt_vocab, tgt_pad_idx, device):
    batch_size = src.shape[0]
    target_sentences_tokens = [[BOS_TOKEN] for _ in range(batch_size)]
    trg_token_ids_batch = torch.tensor([[tgt_vocab[tokens[0]]] for tokens in target_sentences_tokens], device=device)
    is_decoded = [False] * batch_size

    tgt_itos = tgt_vocab.get_itos()
    while True:
        tgt_mask = generate_tgt_mask(trg_token_ids_batch, tgt_pad_idx)
        enc_output, _ = model.encode(src, src_mask)
        predicted_log_distributions, _, _ = model.decode(trg_token_ids_batch, enc_output, src_mask, tgt_mask)

        num_of_trg_tokens = len(target_sentences_tokens[0])
        predicted_log_distributions = predicted_log_distributions[num_of_trg_tokens - 1::num_of_trg_tokens]

        most_probable_last_token_indices = torch.argmax(predicted_log_distributions, dim=-1).cpu().numpy()
        most_probable_last_token_indices = most_probable_last_token_indices.reshape(-1)

        predicted_words = [tgt_itos[index] for index in most_probable_last_token_indices.tolist()]

        non_decoded_indices = [i for i, decoded in enumerate(is_decoded) if not decoded]

        # Filter non_decoded_indices based on the length of predicted_words
        non_decoded_indices = [i for i in non_decoded_indices if i < len(predicted_words)]

        # Create a new list containing the words from predicted_words corresponding to non_decoded_indices
        predicted_words_filtered = [predicted_words[idx] for idx in non_decoded_indices]

        for non_decoded_idx, predicted_word in zip(non_decoded_indices, predicted_words_filtered):
            target_sentences_tokens[non_decoded_idx].append(predicted_word)

            if predicted_word == EOS_TOKEN:
                is_decoded[non_decoded_idx] = True

        if all(is_decoded) or num_of_trg_tokens == max_len:
            break

        # Filter out the decoded sentences and update the tensors accordingly
        src = src[non_decoded_indices]
        src_mask = src_mask[non_decoded_indices]
        trg_token_ids_batch = trg_token_ids_batch[non_decoded_indices]
        if len(non_decoded_indices) == 0:
            break

        most_probable_last_token_indices_filtered = torch.tensor([tgt_vocab[predicted_words_filtered[idx]] for idx, _ in enumerate(non_decoded_indices)], device=device)

        trg_token_ids_batch = torch.cat((trg_token_ids_batch, most_probable_last_token_indices_filtered.unsqueeze(1)), 1)

    return target_sentences_tokens


def evaluate_metrics(model, data_loader, src_pad_idx, tgt_pad_idx, tgt_vocab, max_len, device):
    model.eval()
    bleu_scores = []
    tgt_itos = tgt_vocab.get_itos()
    
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask = generate_src_mask(src, src_pad_idx)
            output = batch_greedy_decode_v2(model, src, src_mask, max_len, tgt_vocab, tgt_pad_idx, device)
            
            hypothesis = [sent[1:-1] for sent in output]  # Remove BOS and EOS tokens
            
            tgt_token_lists = [[tgt_itos[token_idx] for token_idx in sent if token_idx not in (tgt_pad_idx, tgt_vocab[BOS_TOKEN], tgt_vocab[EOS_TOKEN])] for sent in tgt.cpu().numpy()]
            for hyp, ref in zip(hypothesis, tgt_token_lists):
                bleu = calculate_bleu(ref, hyp)
                bleu_scores.append(bleu)
    
    model.train()
    return sum(bleu_scores) / len(bleu_scores)


