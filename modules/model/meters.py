import logging
from collections import defaultdict

import numpy as np
from sklearn import metrics

logger = logging.getLogger(__name__)


class AverageMeter:
    def __init__(self):
        self._counter = 0
        self._avg_value = 0

    def __call__(self):
        return self._avg_value

    def update(self, value):
        self._counter += 1
        self._avg_value = (self._avg_value * (self._counter - 1) + value) / self._counter


class APMeter:
    def __init__(self):
        self.reset()

    def __call__(self):
        return metrics.average_precision_score(self.true_labels,
                                               self.pred_probas)

    def update(self, pred_probas, true_labels):
        self.pred_probas.extend(pred_probas)
        self.true_labels.extend(true_labels)

    def reset(self):
        self.pred_probas = []
        self.true_labels = []


class MAPMeter:
    def __init__(self):
        self.reset()

    def __call__(self):
        metrics = {k: v() for k, v in self.aps_dict.items()}
        metrics['map'] = np.mean(list(metrics.values()))
        return metrics

    def update(self, keys, pred_probas, true_labels):
        assert len(keys) == pred_probas.shape[-1]

        for i, key in enumerate(keys):
            self.aps_dict[key].update(pred_probas[:, i], true_labels == i)

    def reset(self):
        self.aps_dict = defaultdict(APMeter)
