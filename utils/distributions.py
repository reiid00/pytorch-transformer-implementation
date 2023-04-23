import torch
import torch.nn as nn

# Explanation to why KL Divergence Loss
# In "Attention is all you Need", the paper describes the use
# of label smoothing, which inherently involves comparing two probability
# distributions (true distribution and models). KL Divergence is a popular
# choice for this context as it measures the dissimilarity between two probability distributions

# Paper value for label_smoothing: 0.1
class LabelSmoothingKLDivergenceLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        super(LabelSmoothingKLDivergenceLoss, self).__init__()
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing
        self.tgt_vocab_size = tgt_vocab_size
        self.ignore_index = ignore_index
        self.kl_div_loss = nn.KLDivLoss(reduction='none')

    def forward(self, output, target):

        # reshape the output to be (batch_size*seq_len, tgt_vocab_size)
        output = output.view(-1, self.tgt_vocab_size)

        # reshape the target to be (batch_size*seq_len)
        target = target.contiguous().view(-1)

        # Create mask to ignore padding tokens in the target sequence
        # When computing the loss
        non_pad_mask = (target != self.ignore_index).float()

        # Copy output tensor, will store the true distribution after label smoothing
        true_dist = output.clone()

        # Fill the distribution tensor with smoothed probability value to all
        # incorrect classes
        true_dist.fill_(self.label_smoothing / (self.tgt_vocab_size - 1))

        # Update the true distribution tensor with confidence value for the
        # correct classes
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # Calculating KL Div Loss by Hand, can be handful if we want to change the loss function
        # loss = -(true_dist * output).sum(dim=1)
        loss = self.kl_div_loss(output, true_dist.detach()).sum(dim=1)

        # Apply non_padding mask to the loss and compute the sum of the loss values
        loss = (loss * non_pad_mask).sum()

        # Divide by the sum of the non_padding mask to get average loss per token
        loss = loss / non_pad_mask.sum()

        return true_dist, loss # For analysing the distribution.