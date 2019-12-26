import logging
from collections import defaultdict
import os

import torch
import torch.nn as nn
# from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm
import numpy as np

logger = logging.getLogger(__file__)


class AverageMeter:
    def __init__(self):
        self._counter = 0
        self._avg_value = 0

    def __call__(self):
        return self._avg_value

    def update(self, value):
        self._counter += 1
        self._avg_value = (self._avg_value * (self._counter - 1) + value) / self._counter


class Trainer:
    def __init__(self, model, tokenizer, train_dataset, test_dataset, writer, *,
                 device=torch.device('cuda'),
                 train_batch_size=32,
                 test_batch_size=32,
                 batch_split=1,
                 n_jobs=4,
                 n_epochs=0,
                 lr=1e-3,
                 weight_decay=5e-4):

        logger.info(f'Used device: {device}.')

        self.model = model.to(device)

        optimizer_parameters = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in optimizer_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in optimizer_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)

        logger.info(f'Train Dataset len: {len(train_dataset)}.')
        logger.info(f'Test Dataset len: {len(test_dataset)}.')

        train_sampler = RandomSampler(train_dataset)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=int(train_batch_size // batch_split),
                                                            num_workers=n_jobs,
                                                            sampler=train_sampler,
                                                            drop_last=True,
                                                            collate_fn=self.collate_fun)

        self.test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                           batch_size=test_batch_size,
                                                           num_workers=n_jobs,
                                                           shuffle=False,
                                                           drop_last=False,
                                                           collate_fn=self.collate_fun)
        self.device = device
        self.batch_split = batch_split
        self.n_epochs = n_epochs
        self.tokenizer = tokenizer
        self.writer = writer
        self.global_step = 0

    def collate_fun(self, items):
        batch_size = len(items)
        pad_token_id = self.tokenizer.pad_token_id

        max_len = max([len(item.input_ids) for item in items])
        tokens = pad_token_id * np.ones((batch_size, max_len), dtype=np.int64)
        token_type_ids = np.ones((batch_size, max_len), dtype=np.int64)

        for i, item in enumerate(items):
            row = item.input_ids

            tokens[i, :len(row)] = row
            token_type_id = [0 if i <= row.index(102) else 1
                             for i in range(len(row))]  # 102 corresponds to [SEP]

            token_type_ids[i, :len(row)] = token_type_id

        attention_mask = tokens > 0
        inputs = [torch.from_numpy(tokens).to(self.device),
                  torch.from_numpy(attention_mask).to(self.device),
                  torch.from_numpy(token_type_ids).to(self.device)]

        # output labels
        start_ids = np.array([item.start_id for item in items])
        end_ids = np.array([item.end_id for item in items])

        label_ids = [item.label_id for item in items]

        #     start_positions = np.where(start_positions >= max_len, -1, start_positions)
        #     end_positions = np.where(end_positions >= max_len, -1, end_positions)

        labels = [torch.LongTensor(start_ids).to(self.device),
                  torch.LongTensor(end_ids).to(self.device),
                  torch.LongTensor(label_ids).to(self.device)]

        return [inputs, labels]

    def _loss(self, preds, labels):
        start_preds, end_preds, class_preds = preds
        start_labels, end_labels, class_labels = labels

        start_loss = nn.CrossEntropyLoss(ignore_index=-1)(start_preds, start_labels)
        end_loss = nn.CrossEntropyLoss(ignore_index=-1)(end_preds, end_labels)
        class_loss = nn.CrossEntropyLoss()(class_preds, class_labels)

        return start_loss + end_loss + class_loss

    def train(self, after_epoch_funcs=None):
        after_epoch_funcs = [] if after_epoch_funcs is None else after_epoch_funcs

        def run_after_funcs():
            for func in after_epoch_funcs:
                func(epoch_i)

        for epoch_i in range(1, self.n_epochs+1):
            self._train(epoch_i)
            run_after_funcs()

    def _train(self, epoch_i):
        self.model.train()
        self.optimizer.zero_grad()

        avg_losses = defaultdict(AverageMeter)

        tqdm_data = tqdm(self.train_dataloader, desc=f'Train (epoch #{epoch_i} / {self.n_epochs})')

        for i, ((input_ids, attention_mask, token_types), labels) in enumerate(tqdm_data):

            pred_logits = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_types)
            loss = self._loss(pred_logits, labels)
            avg_losses['loss'].update(loss.item())

            loss = loss / self.batch_split
            loss.backward()

            if (i + 1) % self.batch_split == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                for k, v in avg_losses.items():
                    self.writer.add_scalar(f'train/{k}', v(), global_step=self.global_step)

                self.global_step += 1

            tqdm_data.set_postfix({k: v() for k, v in avg_losses.items()})

    def test(self, epoch_i):
        self._test(epoch_i)

    def _test(self, epoch_i):
        self.model.eval()

        avg_losses = defaultdict(AverageMeter)

        tqdm_data = tqdm(self.test_dataloader, desc=f'Test (epoch #{epoch_i} / {self.n_epochs})')

        for i, ((input_ids, attention_mask, token_types), labels) in enumerate(tqdm_data):

            pred_logits = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_types)
            loss = self._loss(pred_logits, labels)
            avg_losses['loss'].update(loss.item())

            tqdm_data.set_postfix({k: v() for k, v in avg_losses.items()})

        for k, v in avg_losses.items():
            self.writer.add_scalar(f'test/{k}', v(), global_step=self.global_step)

    def save_state_dict(self, path_):
        model_dict = self.model.state_dict()

        state_dict = {'model': model_dict,
                      'global_step': self.global_step}

        torch.save(state_dict, path_)

        logger.info(f'State dict was saved to {path_}.')

    def load_state_dict(self, path_):
        if os.path.exists(path_):
            state_dict = torch.load(path_, map_location=self.device)

            self.global_step = state_dict['global_step']
            self.model.load_state_dict(state_dict['model'])

            logger.info(f'Model weights were loaded from {path_} checkpoint.')
