import torch
import torch.nn as nn


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        loss = self.criterion(inputs, targets)
        probs = torch.sigmoid(inputs)

        return torch.mean(self.alpha * (1 - probs)**self.gamma * loss)


class FocalLossWithLogits(nn.Module):

    def __init__(self, alpha=1, gamma=2, *, ignore_index=-1, reduction='mean'):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

        self.criterion = nn.NLLLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, targets):
        log_probs = torch.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        return self.criterion(self.alpha * (1 - probs)**self.gamma * log_probs, targets)
