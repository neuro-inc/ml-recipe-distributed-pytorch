import torch
import torch.nn as nn


class LabelSmoothingLossWithLogits(nn.Module):
    def __init__(self, n_classes, *, smoothing=0.0, ignore_index=-100):
        super().__init__()

        assert 0 <= smoothing <= 1

        self.ignore_index = ignore_index
        self.n_classes = n_classes

        self.smoothing = smoothing
        self.confidence = 1 - smoothing

        self.num_ignore_ixs = 1 + (0 <= self.ignore_index < self.n_classes)

        self.criterion = nn.KLDivLoss(reduction='batchmean') if smoothing > 0 else nn.NLLLoss(ignore_index=ignore_index)

    def forward(self, input_logits, targets):

        log_probas = torch.log_softmax(input_logits, dim=-1)

        if self.smoothing > 0:
            batch_size = targets.size(0)
            fill_value = self.smoothing / (self.n_classes - self.num_ignore_ixs)

            with torch.no_grad():
                target_dist = torch.full((batch_size, self.n_classes), fill_value=fill_value,
                                         device=targets.device)
                target_dist.scatter_(-1, targets.unsqueeze(-1), self.confidence)
                if 0 <= self.ignore_index < self.n_classes:
                    target_dist[:, self.ignore_index] = 0

            targets = target_dist

        return self.criterion(log_probas, targets)


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
