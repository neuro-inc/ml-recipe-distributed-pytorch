import logging
from collections import defaultdict
from dataclasses import dataclass

import torch
# from torch.utils.data import DataLoader
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
        # init_losses = {'start_class': (nn.CrossEntropyLoss(ignore_index=-1), params.w_start),
        #                'end_class': (nn.CrossEntropyLoss(ignore_index=-1), params.w_end),
        #                'start_reg': (nn.MSELoss(), params.w_start_reg),
        #                'end_reg': (nn.MSELoss(), params.w_end_reg),
        #                'cls': (FocalLossWithLogits(alpha=params.focal_alpha, gamma=params.focal_gamma) if params.focal
        #                        else nn.CrossEntropyLoss(weight=train_weights['label_weights']), params.w_cls)}

        keys_ = ['start_class', 'end_class', 'start_reg', 'end_reg', 'cls']

        self.model.eval()

        # async_dataset = DataLoader(dataset,
        #                            batch_size=self.batch_size,
        #                            shuffle=True,
        #                            num_workers=self.n_jobs,
        #                            collate_fn=self.collate_fun)

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
            start_true, end_true, start_reg_true, end_reg_true, cls_true = [labels[k] for k in keys_]

            start_logits, start_ids = torch.max(start_preds, dim=-1)
            end_logits, end_ids = torch.max(end_preds, dim=-1)

            cls_probas, cls_ids = torch.max(torch.softmax(cls_preds, dim=-1), dim=-1)

            # todo: juking
            cls_probas[cls_true != cls_ids] = -1

            # todo: score from paper https://arxiv.org/pdf/1901.08634.pdf
            scores = start_logits + end_logits - (start_preds[:, 0] + end_preds[:, 0])

            # scores[cls_true != cls_ids] = -1

            self._update_candidates(scores.numpy(),
                                    start_ids.numpy(), end_ids.numpy(),
                                    start_reg_preds.numpy(), end_reg_preds.numpy(),
                                    cls_ids.numpy(), items)

            if save_dump:
                self.dump.append((scores.numpy(), start_ids.numpy(), end_ids.numpy(), cls_ids.numpy(), items))

            if self.limit is not None and batch_i >= self.limit:
                break

    def show_predictions(self):
        for item_id, score in self.scores.items():
            logger.info(20 * '=')

            if item_id in self.items and item_id in self.candidates:
                item = self.items[item_id]

                candidate = self.candidates[item_id]

                start_id = candidate.start_id
                end_id = candidate.end_id
                # (start_id, end_id, label) = self.candidates[item_id]

                true_start = item.true_start
                true_end = item.true_end

                if RawPreprocessor.id2labels[candidate.label] in ['short', 'long']:
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
                            f'{(start_id, end_id, RawPreprocessor.id2labels[candidate.label])} | '
                            f'{(true_start, true_end, RawPreprocessor.id2labels[item.true_label])}')

                logger.info(f'start_reg: {candidate.start_reg}, end_reg: {candidate.end_reg}')

            else:
                logger.warning(f'Something wrong with {item_id}.')
        # self.scores = defaultdict(int)
        # self.candidates = {}
        # self.items = {}
