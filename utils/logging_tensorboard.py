from torch.utils.tensorboard import SummaryWriter
from utils.visualize import *

def create_summary_writer(log_dir):
    return SummaryWriter(log_dir=log_dir)

def log_loss(writer, loss, global_step):
    writer.add_scalar("Loss/train", loss.item(), global_step)

def log_learning_rate(writer, learning_rate, global_step):
    writer.add_scalar("Learning Rate", learning_rate, global_step)

def log_gradients(writer, model, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f"Gradients/{name}", param.grad, global_step)

def log_attention_weights(writer, attention, n_layers, n_heads, global_step):
    img_array = plot_attention_weights_grid(attention,n_layers, n_heads)
    writer.add_image("Attention Weights Grid", img_array, global_step, dataformats="HWC")
