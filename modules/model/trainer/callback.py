import logging
import math

import torch
from sklearn import metrics

from .meters import MAPMeter, AverageMeter

logger = logging.getLogger(__name__)


class TestCallback:
    def at_iteration_end(self, preds, labels, avg_meters):
        self._at_iteration_end(preds, labels, avg_meters)

    def _at_iteration_end(self, *args):
        raise NotImplemented

    def at_epoch_end(self, avg_meters, trainer):
        self._at_epoch_end(avg_meters, trainer)
        self._reset()

    def _at_epoch_end(self, *args):
        raise NotImplemented

    def _reset(self):
        pass


class AccuracyCallback(TestCallback):
    keys = ['start_class', 'end_class', 'cls']

    def __init__(self):
        super().__init__()

    def _at_iteration_end(self, preds, labels, avg_meters):
        start_logits, end_logits, cls_logits = (preds[k].detach().cpu() for k in self.keys)
        start_true, end_true, cls_true = (labels[k].detach().cpu() for k in self.keys)

        start_pred, end_pred, cls_pred = (torch.max(logits, dim=-1)[1] for logits in
                                          (start_logits, end_logits, cls_logits))

        start_idxs = start_true != -1
        end_idxs = end_true != -1

        if any(start_idxs):
            avg_meters['s_acc'].update(metrics.accuracy_score(start_true[start_idxs], start_pred[start_idxs]))
        if any(end_idxs):
            avg_meters['e_acc'].update(metrics.accuracy_score(end_true[end_idxs], end_pred[end_idxs]))
        avg_meters['c_acc'].update(metrics.accuracy_score(cls_true, cls_pred))

    def _at_epoch_end(self, *args):
        pass


class MAPCallback(TestCallback):
    key = 'cls'

    def __init__(self, metric_keys):
        super().__init__()
        self._metric_keys = metric_keys
        self._reset()

    def _at_iteration_end(self, preds, labels, *args):
        cls_logits = preds[self.key].detach().cpu()
        cls_true = labels[self.key].detach().cpu()

        self.map_meter.update(keys=self._metric_keys,
                              pred_probas=torch.softmax(cls_logits, dim=-1).numpy(),
                              true_labels=cls_true.numpy())

    def _at_epoch_end(self, avg_meters, *args):
        avg_meters.update(self.map_meter())

    def _reset(self):
        self.map_meter = MAPMeter()


class SaveBestCallback(TestCallback):
    def __init__(self, params):
        super().__init__()

        self.params = params

        self.metric = self.params.best_metric
        self.best_order = self.params.best_order

        self.value = 1e10 * (-1 if self.best_order == '>' else 1)

    def _at_iteration_end(self, *args):
        pass

    def _at_epoch_end(self, avg_meters, trainer):
        metrics = {k: v() if isinstance(v, AverageMeter) else v for k, v in avg_meters.items()}

        if self.metric in metrics and not math.isnan(metrics[self.metric]):

            if eval(f'{metrics[self.metric]}{self.best_order}{self.value}'):
                self.value = metrics[self.metric]
                trainer.save_state_dict(self.params.dump_dir / self.params.experiment_name / f'best.ch')

                logger.info(f'Best value of {self.metric} was achieved after training step {trainer.global_step} '
                            f'and equals to {self.value:.3f}')
            else:
                logger.info(f'Best value {self.value:.3f} of {self.metric} was not bitten '
                            f'with {metrics[self.metric]:.3f}')
        else:
            logger.warning(f'Trainer metrics do not contain metric {self.metric}.')


