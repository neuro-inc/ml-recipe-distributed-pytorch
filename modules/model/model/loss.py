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
        probs = torch.exp(-loss)

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


class WeightedLoss:
    def __init__(self, init_losses):
        self._losses = init_losses

    def __call__(self, preds, targets, *, avg_meters=None):
        assert set(preds.keys()).intersection(set(targets.keys())) == set(preds.keys())
        assert set(self._losses.keys()).intersection(set(targets.keys())) == set(self._losses.keys())

        full_loss = 0

        for key in self._losses.keys():
            pred = preds[key]
            target = targets[key]

            loss_f, weight = self._losses[key]

            loss = loss_f(pred, target)

            if avg_meters is not None:
                avg_meters[key] = loss.item()

            full_loss += weight * loss

        if avg_meters is not None:
            avg_meters['loss'] = full_loss.item()

        return full_loss

    def to(self, device):
        for key in self._losses.keys():
            self._losses[key][0].to(device)

        return self
