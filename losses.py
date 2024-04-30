import torch
from torch.nn import functional as F

class TransLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TransLoss, self).__init__()
        self.margin = margin

    def forward(self, positive_scores, negative_scores):
        # Calculate the margin-based ranking loss
        loss = F.relu(self.margin + positive_scores - negative_scores)
        return torch.mean(loss)