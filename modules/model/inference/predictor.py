import logging
from collections import defaultdict
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from .. utils.list_dataloader import ListDataloader
from .. dataset import RawPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class PredictorCandidate:
    start_id: int
    end_id: int
    start_reg: int
    end_reg: int
    label: int


class Predictor:
    def __init__(self, model, device, *,
                 batch_size=256,
                 n_jobs=16,
                 collate_fun=None,
                 buffer_size=4096,
                 limit=None):
        self.model = model
        self.device = device

        self.model.to(device)

        self.scores = defaultdict(int)
        self.candidates = {}
        self.items = {}

        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.collate_fun = collate_fun
        self.buffer_size = buffer_size

        self.limit = limit

        self.dump = None

        logger.info(f'Predictor uses {self.device} device. Batch size: {self.batch_size}. '
                    f'#workers: {self.n_jobs}. Buffer size: {self.buffer_size}. Set limit: {self.limit}.')

    def _to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [self._to_device(d) for d in data]
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        elif data is None:
            return data
        else:
            raise NotImplemented

    def _is_valid(self, item, score, start_id, end_id):
        assert score >= 0

        if start_id > end_id:
            return False

        if start_id < item.question_len + 2:
            return False

        if self.scores[item.item_id] > score:
            return False

        return True

    def _update_candidates(self, scores, start_ids, end_ids, start_regs, end_regs, labels, items):
        for score, start_id, end_id, srart_reg, end_reg, label, item in zip(scores, start_ids, end_ids, start_regs, end_regs, labels, items):
            if self._is_valid(item, score, start_id, end_id):
                self.scores[item.item_id] = score

                self.candidates[item.item_id] = PredictorCandidate(start_id=start_id,
                                                                   end_id=end_id,
                                                                   start_reg=srart_reg,
                                                                   end_reg=end_reg,
                                                                   label=label)
                self.items[item.item_id] = item

    @torch.no_grad()
    def __call__(self, dataset, *, save_dump=False):
        keys_ = ['start_class', 'end_class', 'start_reg', 'end_reg', 'cls']

        self.model.eval()

        async_dataset = ListDataloader(dataset,
                                       batch_size=self.batch_size,
                                       n_jobs=self.n_jobs,
                                       collate_fun=self.collate_fun,
                                       buffer_size=self.buffer_size,
                                       shuffle=True)

        if save_dump:
            self.dump = []

        tqdm_data = tqdm(async_dataset, desc='Processing documents. It can take a while', total=self.limit)
        for batch_i, (inputs, labels, items) in enumerate(tqdm_data):
            inputs = self._to_device(inputs)

            preds = self.model(**inputs)

            start_preds, end_preds, start_reg_preds, end_reg_preds, cls_preds = [preds[k].detach().cpu() for k in keys_]
            # start_true, end_true, start_reg_true, end_reg_true, cls_true = [labels[k] for k in keys_]

            start_logits, start_ids = torch.max(start_preds, dim=-1)
            end_logits, end_ids = torch.max(end_preds, dim=-1)

            cls_probas, cls_ids = torch.max(torch.softmax(cls_preds, dim=-1), dim=-1)

            # score from paper https://arxiv.org/pdf/1901.08634.pdf
            scores = start_logits + end_logits - (start_preds[:, 0] + end_preds[:, 0])

            self._update_candidates(scores.numpy(),
                                    start_ids.numpy(), end_ids.numpy(),
                                    start_reg_preds.numpy(), end_reg_preds.numpy(),
                                    cls_ids.numpy(), items)

            if save_dump:
                self.dump.append((scores.numpy(), start_ids.numpy(), end_ids.numpy(), cls_ids.numpy(), items))

            if self.limit is not None and batch_i >= self.limit:
                break

    def show_predictions(self, *, n_docs=None):
        for doc_i, doc_id in enumerate(self.scores.keys()):
            if n_docs is not None and doc_i >= n_docs:
                break

            doc = self.items[doc_id]
            candidate = self.candidates[doc_id]

            logger.info(f'Text: {doc.true_text}')
            logger.info(f'Question: {doc.true_question}')
            logger.info(f'True label: {RawPreprocessor.id2labels[doc.true_label]}. '
                        f'Pred label: {RawPreprocessor.id2labels[candidate.label]}.')
