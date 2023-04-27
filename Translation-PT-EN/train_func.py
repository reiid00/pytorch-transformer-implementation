import os
import sys
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(f'..{os.sep}utils'))))
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname( '..'))))
from utils.constants import *
import torch
import torch.nn as nn
from transformer_v2 import Transformer
from utils.function_utils import *
from func_load_model import *
from utils.logging_tensorboard import create_summary_writer, log_loss, log_learning_rate, log_gradients
from utils.optimizer_n_scheduler import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
max_len = MODEL_MAX_SEQ_LEN
d_model = MODEL_DIM
num_layers = MODEL_N_LAYERS
num_heads = MODEL_N_HEADS
dropout = MODEL_DROPOUT
num_epochs = 10
learning_rate = 1e-4
warmup_steps = 2000
weight_decay = 1e-4
VOCAB_SIZE = 32_000
d_ff = MODEL_FF

NUM_PHRASES = 1_000_000

n=0
LOGGING_FILE = f'runs{os.sep}translation_experiment_{n}'

tokenizer = load_tokenizer()
model = Transformer(VOCAB_SIZE,
                    VOCAB_SIZE, 
                    d_model, 
                    num_heads, 
                    num_layers, 
                    d_ff, 
                    dropout, 
                    max_len).to(device)

optimizer, scheduler = create_optimizer_and_scheduler(model, d_model, warmup_steps, learning_rate, weight_decay)

# Initialize TensorBoard SummaryWriter
writer = create_summary_writer(LOGGING_FILE)

sentence_pairs = load_dataset(FILE_PATH, limit=NUM_PHRASES)
split_idx = int(len(sentence_pairs) * 0.9)
train_sentence_pairs = sentence_pairs[:split_idx]
val_sentence_pairs = sentence_pairs[split_idx:]

train_dataset = preprocess_data(train_sentence_pairs, tokenizer, max_len)
val_dataset = preprocess_data(val_sentence_pairs, tokenizer, max_len)

train_dataloader = create_dataloader(train_dataset, batch_size, tokenizer, shuffle=True, num_workers=0)
val_dataloader = create_dataloader(val_dataset, batch_size, tokenizer, shuffle=False, num_workers=0)

global_step = 0
def train_model():
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (source, target) in enumerate(train_dataloader):
            source, target = source.to(device), target.to(device)

            # Forward pass
            output = model(source, target[:-1, :])
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            # Calculate loss
            loss = nn.CrossEntropyLoss(ignore_index=0)(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            log_gradients(writer, model, global_step)  # Log gradients to TensorBoard
            optimizer.step()
            scheduler.step()

            # Log loss and learning rate to TensorBoard
            log_loss(writer, loss, global_step)
            log_learning_rate(writer, scheduler.learning_rate(), global_step)
            
            global_step += 1

        # Evaluate model on the validation set
        model.eval()
        bleu_score = calculate_bleu(val_dataloader, model, tokenizer)
        print(f"Epoch [{epoch+1}/{num_epochs}], BLEU Score: {bleu_score:.4f}")
        writer.add_scalar("BLEU Score/validation", bleu_score, epoch)

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "bleu_score": bleu_score,
        }
        save_checkpoint(checkpoint, f"checkpoints/translation_epoch{epoch+1}.pth")

    writer.close()