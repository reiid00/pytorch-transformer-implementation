import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()

        # Embeds source/target token ids into embedding vectors
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Adds positional information to token's
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Stacks (num_layers) of Encoders and Decoders
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # final layer to generate output probabilities
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self.__init__params_xavier()
    
    def __init__params_xavier(self):
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        # Apply the input embedding and positional encoding
        src_emb = self.encoder_embedding(src)
        src_emb = self.pos_encoding(src_emb)

        # Pass the input through the encoder layers
        enc_output = src_emb
        enc_attention_weights = []
        for layer in self.encoder_layers:
            enc_output, attn_weights  = layer(enc_output, src_mask)
            enc_attention_weights.append(attn_weights)

        return enc_output, enc_attention_weights

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        # Apply the output embedding and positional encoding
        tgt_emb = self.decoder_embedding(tgt)
        tgt_emb = self.pos_encoding(tgt_emb)

        # Pass the output through the decoder layers with the encoded input
        dec_output = tgt_emb
        dec_attention_weights = []
        dec_cross_attention_weights = []
        for layer in self.decoder_layers:
            dec_output, attn_weights, cross_attention_weights = layer(dec_output, enc_output, src_mask, tgt_mask)
            dec_attention_weights.append(attn_weights)
            dec_cross_attention_weights.append(cross_attention_weights)
        
        return dec_output, dec_attention_weights, dec_cross_attention_weights

    def forward(self, src, tgt, src_mask, tgt_mask, return_attention = False):
        # Encode the input
        enc_output, enc_attention_weights = self.encode(src, src_mask)

        # Decode the output using the encoded input
        dec_output, dec_attention_weights, dec_cross_attention_weights = self.decode(tgt, enc_output, src_mask, tgt_mask)

        # Generate the output probabilities
        output_probs = F.log_softmax(self.generator(dec_output), dim=-1)

        if return_attention:
            return output_probs, enc_attention_weights, dec_attention_weights, dec_cross_attention_weights
        return output_probs

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def masked_self_attention(self, x, tgt_mask):
        # Perform (Masked) self-attention on the input
        attn_output, attn_weights = self.self_attn(x, x, x, tgt_mask)

        # Add & Norm (Masked) masked_self_attention output + "x", input, in this case, output
        return self.layer_norm1(x + self.dropout(attn_output)), attn_weights

    def cross_attention(self, x, enc_output, src_mask):
        # Perform cross-attention using the encoded input
        attn_output, cross_attn_weights = self.cross_attn(x, enc_output, enc_output, src_mask)

        # Add & Norm cross-attention output +  + layer_norm1 output
        return self.layer_norm2(x + self.dropout(attn_output)), cross_attn_weights

    def positionwise_feedforward(self, x):
        # Apply the position-wise feed-forward network
        ff_output = self.feed_forward(x)

        # Add & Norm feed-forward output +  + layer_norm2 output
        return self.layer_norm3(x + self.dropout(ff_output))

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Apply masked_self_attention + Add & Norm
        x, attn_weights = self.masked_self_attention(x, tgt_mask)

        # Apply cross-attention with the encoded input + Add & Norm
        x, cross_attn_weights = self.cross_attention(x, enc_output, src_mask)

        # Apply position-wise feed-forward network + Add & Norm
        x = self.positionwise_feedforward(x)

        return x, attn_weights, cross_attn_weights

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def self_attention(self, x, mask):

        # Perform self-attention on the input
        attn_output, attn_weights = self.self_attn(x, x, x, mask)

        # Add & Norm self-attention output + "x", input
        return self.layer_norm1(x + self.dropout(attn_output)), attn_weights

    def positionwise_feedforward(self, x):

        # Apply the position-wise feed-forward network
        ff_output = self.feed_forward(x)

        # Add & Norm Feed-forward output + layer_norm1 output
        return self.layer_norm2(x + self.dropout(ff_output))

    def forward(self, x, mask):
        # Apply self-attention + Add & Norm
        x, attn_weights= self.self_attention(x, mask)

        # Apply position-wise feed-forward network + Add & Norm
        x = self.positionwise_feedforward(x)

        return x, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads # Dimensions of keys (which will be equal to queries and values)
        self.num_heads = num_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # Original shape of x : (batch_size, seq_len, d_model)
        # We want:
        # (batch_size, num_heads, seq_len, d_k)

        # We get batch_size and seq_len
        batch_size, seq_len, _ = x.shape

        # Reshape x to be (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose to get dimensions (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, q, k, v, mask):

        # Calculate attention scores

        # Matmul between q.k

        # q shape (batch_size, num_heads, d_k, seq_len)
        # k shape (batch_size, num_heads, d_k, seq_len)

        # Wouldn't work if we calculated the dot product like this
        # num_columns q (d_k) =/= num_rows k (seq_len)

        # If we Transpose
        # We get k (batch_size, num_heads, seq_len, d_k)
        # Then, num_columns q (d_k) == num_rows k (d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        # attn_scores shape = (batch_size, num_heads, seq_len, seq_len)
        # Which represents the pairwise attention scores between tokens

        # Scaling the scores
        # This ensure the dot product remain in a reasonable range
        # Gradients backpropagation will be more stable
        attn_scores = attn_scores / math.sqrt(self.d_k)

        # Masking (Optional)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-1e20'))

        # Softmax is applied to the attention scores
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Multiply weights by the values
        return torch.matmul(attn_weights, v), attn_weights

    def forward(self, q, k, v, mask=None):

        batch_size = q.size(0)

        # Pass the inputs through the linear layers  and into multiple heads
        # Basically the first part of multi-head Attention
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Perform scaled dot-product attention
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # Concatenate the multi-head outputs
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Pass the concatenated output through the final linear layer
        return self.linear_out(attn_output), attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Tensor filled with 0s
        # Will store positional encodings for each position in the sequence
        pe = torch.zeros(max_len, d_model)
        
        # Tensor, shape (max_len, 1)
        # Contains position indices from 0 to max_len -1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Tensor, shape (d_model //2,)
        # Contains series of values used to scale the positional indices in the sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # Sine element-wise product between position and div_term
        # Assign the result to the even indices of pe
        pe[:, 0::2] = torch.sin(position * div_term)

        # Cosine element-wise product between position and div_term
        # Assign the result to the odd indices of pe
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add extra dimension to pe to match input tensor's shape
        # We had (max_len, d_model)
        # After unsequeeze 0
        # (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Strictly to register the pe as a persistent buffer in the module
        # Allows to save and load the tensor with module's state_dict
        self.register_buffer('pe', pe)

    def forward(self, x):
        # self.pe[:, :x.size(1)] slices pe to match length of input sequence
        # pe : (1,max_len, d_model)
        # After slicing
        # pe : (1, seq_len, d_model)

        # x + self.pe[:, :x.size(1)] Adds the position encoding tensor to x
        # Since they have the same shapes, the addition can be done element-wise
        # x.shape : (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x
    
    def get_encodings(self):
        encodings = self.pe.squeeze(0)
        return encodings.cpu().numpy()

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))