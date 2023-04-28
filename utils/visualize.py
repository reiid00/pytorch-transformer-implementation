import matplotlib.pyplot as plt
import torch
import sys
import os
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname( '..'))))
from transformer_v2 import Transformer
from utils.function_utils import *
import numpy as np
import io
from PIL import Image

def plot_positional_encodings(encodings, outfile=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(encodings, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Positional Encodings")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Token Position")
    if outfile is not None:
        plt.savefig(outfile)
    plt.show()

def plot_attention_weights_layer_head(attention, layer, head, outfile=None, print_img = True):
    attention_data = attention[layer][0, head].cpu().detach().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(attention_data, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f"Attention Weights (Layer: {layer}, Head: {head})")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    if outfile is not None:
        plt.savefig(outfile)
    if print_img:
        plt.show()

def plot_attention_weights_grid(attention_weights, n_layers, n_heads):
    num_layers = n_layers
    num_heads = n_heads

    grids = ['encoder', 'decoder_self', 'decoder_enc_dec']
    grid_titles = ['Encoder Attention Weights', 'Decoder Self-Attention Weights', 'Decoder Cross-Attention Weights']

    for grid_idx, grid in enumerate(grids):
        fig, axes = plt.subplots(num_layers, num_heads, figsize=(3*num_heads, 3*num_layers), sharex=True, sharey=True)

        for layer in range(num_layers):
            for head in range(num_heads):
                attention_data = attention_weights[grid][layer][0, head].cpu().detach().numpy()
                ax = axes[layer, head]
                img = ax.imshow(attention_data, cmap='viridis', aspect='auto')
                ax.set_title(f"L{layer+1}, H{head+1}")

        fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.7)
        fig.suptitle(grid_titles[grid_idx])
        fig.text(0.5, 0.04, "Key Positions", ha="center", va="center")
        fig.text(0.04, 0.5, "Query Positions", ha="center", va="center", rotation="vertical")

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        plt.close()

    return img_array

def view_attention_weights(model, src_text, tgt_text):

    src = src_text
    tgt = tgt_text
    src_mask, tgt_mask = generate_masks(src, tgt)
    with torch.no_grad():
        _, enc_attention_weights, dec_self_attention_weights, dec_enc_attention_weights = model(src, tgt, src_mask, tgt_mask, return_attention=True)
    layer = 2
    head = 1
    outfile = f"enc_attention_weights_layer{layer}_head{head}.png"
    plot_attention_weights_layer_head(enc_attention_weights, layer, head, outfile)

    outfile = f"dec_self_attention_weights_layer{layer}_head{head}.png"
    plot_attention_weights_layer_head(dec_self_attention_weights, layer, head, outfile)

    outfile = f"dec_enc_attention_weights_layer{layer}_head{head}.png"
    plot_attention_weights_layer_head(dec_enc_attention_weights, layer, head, outfile)

def plot_learning_rate_decay(scheduler, num_steps, outfile=None):
    learning_rates = []
    for step in range(1, num_steps + 1):
        scheduler.current_step = step
        learning_rates.append(scheduler.learning_rate())

    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates)
    plt.title("Learning Rate Decay")
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.grid()
    if outfile is not None:
        plt.savefig(outfile)
    plt.show()

def plot_grad_flow(named_parameters):
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())

    plt.figure(figsize=(30, 10))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="salmon", label="Max gradient")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="skyblue", label="Mean gradient")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    
    ax = plt.gca()
    ax.set_xticks(range(0, len(ave_grads), 1), minor=False)
    ax.set_xticklabels(layers, rotation="vertical", fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(ave_grads) + 0.5, 1), minor=True)
    ax.tick_params(axis='x', which='minor', length=0)

    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=None)
    plt.xlabel("Layers")
    plt.ylabel("Gradient values")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()