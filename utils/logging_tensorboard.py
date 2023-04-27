from torch.utils.tensorboard import SummaryWriter

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

def log_attention_weights(writer, attention, layer, head, global_step):
    writer.add_image(f"Attention Weights/Layer {layer}/Head {head}", attention, global_step, dataformats="HW")
