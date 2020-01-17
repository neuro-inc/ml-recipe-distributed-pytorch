import logging
from collections import defaultdict

import torch
from tqdm.auto import tqdm

from .async_dataloader import AsyncDatasetProcessor
from .split_dataset import RawPreprocessor

logger = logging.getLogger(__name__)


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
        else:
            raise NotImplemented

    def _is_valid(self, item, score, start_id, end_id):
        if start_id > end_id:
            return False

        if start_id < item.question_len + 2:
            return False

        if self.scores[item.item_id] > score:
            return False

        return True

    def _update_candidates(self, scores, start_ids, end_ids, labels, items):
        for score, start_id, end_id, label, item in zip(scores, start_ids, end_ids, labels, items):
            if self._is_valid(item, score, start_id, end_id):
                self.scores[item.item_id] = score
                self.candidates[item.item_id] = (start_id, end_id, label)
                self.items[item.item_id] = item

    @torch.no_grad()
    def __call__(self, dataset, *, save_dump=False):
        self.model.eval()

        async_dataset = AsyncDatasetProcessor(dataset,
                                              batch_size=self.batch_size,
                                              n_jobs=self.n_jobs,
                                              collate_fun=self.collate_fun,
                                              buffer_size=self.buffer_size,
                                              shuffle=True)

        if save_dump:
            self.dump = []

        tqdm_data = tqdm(async_dataset, desc='Processing documents. It can take a while', total=self.limit)
        for batch_i, (inputs, labels, items) in enumerate(tqdm_data):
            (input_ids, attention_mask, token_type_ids) = self._to_device(inputs)

            preds = self.model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)

            start_preds, end_preds, cls_preds = [l.detach().cpu() for l in preds]
            start_true, end_true, cls_true = labels

            start_logits, start_ids = torch.max(start_preds, dim=-1)
            end_logits, end_ids = torch.max(end_preds, dim=-1)

            cls_probas, cls_ids = torch.max(torch.softmax(cls_preds, dim=-1), dim=-1)
            # cls_probas[cls_true != cls_ids] = -1

            scores = start_logits + end_logits - (start_preds[:, 0] + end_preds[:, 0])

            # scores[cls_true != cls_ids] = -1

            self._update_candidates(scores.numpy(), start_ids.numpy(), end_ids.numpy(), cls_ids.numpy(), items)

            if save_dump:
                self.dump.append((scores.numpy(), start_ids.numpy(), end_ids.numpy(), cls_ids.numpy(), items))

            if self.limit is not None and batch_i >= self.limit:
                break

    def show_predictions(self):
        for item_id, score in self.scores.items():
            if item_id in self.items and item_id in self.candidates:
                item = self.items[item_id]
                (start_id, end_id, label) = self.candidates[item_id]

                true_start = item.true_start
                true_end = item.true_end

                if RawPreprocessor.id2labels[label] in ['short', 'long']:
                    start_id += item.chunk_start - item.question_len - 2
                    end_id += item.chunk_start - item.question_len - 2

                    if 0 <= start_id < len(item.t2o):
                        start_id = item.t2o[start_id]
                    else:
                        logger.warning(f'start_id: {start_id} is out of border: {len(item.t2o)}.')

                    if 0 <= end_id < len(item.t2o):
                        end_id = item.t2o[end_id]
                    else:
                        logger.warning(f'end_id: {end_id} is out of border: {len(item.t2o)}.')

                    true_start = item.t2o[item.true_start]
                    true_end = item.t2o[item.true_end]

                    true_text = item.true_text.split()
                    true_answer = ' '.join(true_text[true_start:true_end])
                    pred_answer = ' '.join(true_text[start_id:end_id])

                    logger.info(f'TRUE ANSWER: {true_answer}.')
                    logger.info(f'PRED ANSWER: {pred_answer}.')

                logger.info(f'id: {item_id}. q: {item.true_question}. score: {score}. '
                            f'{(start_id, end_id, RawPreprocessor.id2labels[label])} | '
                            f'{(true_start, true_end, RawPreprocessor.id2labels[item.true_label])}')
            else:
                logger.warning(f'Something wrong with {item_id}.')
        # self.scores = defaultdict(int)
        # self.candidates = {}
        # self.items = {}
