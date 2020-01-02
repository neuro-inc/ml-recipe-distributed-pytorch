import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
# from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__file__)


def initialize_apex(model, optimizer, apex_level=None, apex_verbosity=0):
    if apex_level is not None:
        try:
            from apex import amp
        except ImportError:
            raise ImportError('Install Apex to train model with mixed precision.')

        model, optimizer = amp.initialize(model, optimizer, opt_level=apex_level,
                                          verbosity=apex_verbosity)

    return model, optimizer


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
    def __init__(self, model, tokenizer, train_dataset, test_dataset, writer_dir, *,
                 device=torch.device('cuda'),
                 train_batch_size=32,
                 test_batch_size=32,
                 batch_split=1,
                 n_jobs=4,
                 n_epochs=0,
                 lr=1e-3,
                 weight_decay=5e-4,
                 w_start=1,
                 w_end=1,
                 w_cls=1,
                 warmup_coef=0.01,
                 apex_level=None,
                 apex_verbosity=1,
                 drop_optimizer=False,
                 train_weights=None,
                 debug=False,
                 max_grad_norm=1,
                 local_rank=-1,
                 gpu_id=None,
                 sync_bn=False):

        if sync_bn and local_rank != -1:
            try:
                import apex
                model = apex.parallel.convert_syncbn_model(model)
                logger.info('BatchNorm was synchronized across nodes with APEX module.')
            except ImportError:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                logger.info('BatchNorm was synchronized across nodes with Pytorch module.')

        self.model = model.to(device)

        optimizer_parameters = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in optimizer_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in optimizer_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        # todo: incorrect value during distributed training
        num_training_steps = n_epochs * len(train_dataset) // train_batch_size // batch_split
        num_warmup_steps = num_training_steps * warmup_coef

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps,
                                                         num_training_steps=num_training_steps)

        self.model, self.optimizer = initialize_apex(self.model,  self.optimizer,
                                                     apex_level=apex_level, apex_verbosity=apex_verbosity)

        if local_rank != -1:
            if gpu_id is not None:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=True
                )
            else:
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)

        self.apex_level = apex_level
        self.apex_verbosity = apex_verbosity
        logger.info(f'APEX optimization level: {self.apex_level}. APEX verbosity: {self.apex_verbosity}.')

        logger.info(f'Train Dataset len: {len(train_dataset)}.')
        logger.info(f'Test Dataset len: {len(test_dataset)}.')

        if local_rank == -1:
            train_sampler = RandomSampler(train_dataset) if train_weights is None \
                else WeightedRandomSampler(train_weights, len(train_weights))
        else:
            train_sampler = DistributedSampler(train_dataset)

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

        self.start_loss = nn.CrossEntropyLoss(ignore_index=-1).to(device)
        self.end_loss = nn.CrossEntropyLoss(ignore_index=-1).to(device)
        self.cls_loss = nn.CrossEntropyLoss().to(device)

        self.w_start = w_start
        self.w_end = w_end
        self.w_cls = w_cls

        self.max_grad_norm = max_grad_norm

        self.device = device
        self.local_rank = local_rank
        self.batch_split = batch_split
        self.n_epochs = n_epochs
        self.tokenizer = tokenizer
        self.global_step = 0
        self.debug = debug
        self.drop_optimizer = drop_optimizer

        self.writer = SummaryWriter(log_dir=writer_dir) if self.local_rank in [-1, 0] else None

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

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

        labels = [torch.LongTensor(start_ids).to(self.device),
                  torch.LongTensor(end_ids).to(self.device),
                  torch.LongTensor(label_ids).to(self.device)]

        return [inputs, labels]

    def _loss(self, preds, labels, *, avg_losses=None):
        start_preds, end_preds, cls_preds = preds
        start_labels, end_labels, cls_labels = labels

        start_loss = self.start_loss(start_preds, start_labels)
        end_loss = self.end_loss(end_preds, end_labels)
        cls_loss = self.cls_loss(cls_preds, cls_labels)

        loss = self.w_start * start_loss + self.w_end * end_loss + self.w_cls * cls_loss

        if avg_losses is not None:
            avg_losses['start_loss'].update(start_loss.item())
            avg_losses['end_loss'].update(end_loss.item())
            avg_losses['cls_loss'].update(cls_loss.item())

            avg_losses['loss'].update(loss.item())

        return loss

    def _backward(self, loss):
        if self.apex_level is not None:
            try:
                from apex import amp
            except ImportError:
                raise ImportError('Install Apex to train model with mixed precision.')

            with amp.scale_loss(loss, self.optimizer) as scale_loss:
                scale_loss.backward()
        else:
            loss.backward()

    def train(self, after_epoch_funcs=None):
        after_epoch_funcs = [] if after_epoch_funcs is None else after_epoch_funcs

        def run_after_funcs():
            for func in after_epoch_funcs:
                func(epoch_i)

        for epoch_i in range(1, self.n_epochs+1):
            self._train(epoch_i)
            run_after_funcs()

    @staticmethod
    def _update_console(tqdm_data, losses):
        tqdm_data.set_postfix({k: v() if isinstance(v, AverageMeter) else v for k, v in losses.items()})

    def _update_writer(self, losses, *, prefix=None):
        if self.local_rank in [-1, 0]:
            for k, v in losses.items():
                self.writer.add_scalar(f'{prefix}/{k}', v() if isinstance(v, AverageMeter) else v,
                                       global_step=self.global_step)

    def _clip_grad_norm(self):
        if self.apex_level is not None:
            try:
                from apex import amp
            except ImportError:
                raise ImportError('Install Apex to train model with mixed precision.')

            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    def _train(self, epoch_i):
        self.model.train()
        self.optimizer.zero_grad()

        avg_losses = defaultdict(AverageMeter)

        tqdm_data = tqdm(self.train_dataloader, desc=f'Train (epoch #{epoch_i} / {self.n_epochs})')

        for i, ((input_ids, attention_mask, token_type_ids), labels) in enumerate(tqdm_data):
            pred_logits = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)
            loss = self._loss(pred_logits, labels, avg_losses=avg_losses)

            loss = loss / self.batch_split if self.batch_split > 1 else loss

            self._backward(loss)

            avg_losses['lr'] = self.get_lr()

            if (i + 1) % self.batch_split == 0:
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self._update_writer(avg_losses, prefix='train')

                self.global_step += 1

                if self.debug:
                    logger.info('Training was interrupted because of debug mode.')
                    break

            Trainer._update_console(tqdm_data, avg_losses)

    def test(self, epoch_i):
        if self.local_rank == -1:
            self._test(epoch_i)

    def _test(self, epoch_i):
        self.model.eval()

        avg_losses = defaultdict(AverageMeter)

        tqdm_data = tqdm(self.test_dataloader, desc=f'Test (epoch #{epoch_i} / {self.n_epochs})')

        for i, ((input_ids, attention_mask, token_type_ids), labels) in enumerate(tqdm_data):

            pred_logits = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)

            _ = self._loss(pred_logits, labels, avg_losses=avg_losses)

            start_logits, end_logits, cls_logits = (logits.detach().cpu() for logits in pred_logits)
            start_true, end_true, cls_true = (label.detach().cpu() for label in labels)

            start_pred, end_pred, cls_pred = (torch.max(logits, dim=-1)[1] for logits in (start_logits, end_logits, cls_logits))

            start_idxs = start_true != -1
            end_idxs = end_true != -1

            avg_losses['s_acc'].update(metrics.accuracy_score(start_true[start_idxs], start_pred[start_idxs]))
            avg_losses['e_acc'].update(metrics.accuracy_score(end_true[end_idxs], end_pred[end_idxs]))
            avg_losses['c_acc'].update(metrics.accuracy_score(cls_true, cls_pred))

            # mask = (cls_true == RawDataPreprocessor.labels2id['short']) | (cls_true == RawDataPreprocessor.labels2id['long'])

            Trainer._update_console(tqdm_data, avg_losses)

            if self.debug:
                logger.info('Test was interrupted because of debug mode.')
                break

        self._update_writer(avg_losses, prefix='test')

    def save_state_dict(self, path_):
        if self.local_rank not in [-1, 0]:
            return

        if self.debug:
            logger.info(f'Model was not saved to {path_} because of debug mode.')
            return

        model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

        model_dict = model.state_dict()
        optimizer_dict = self.optimizer.state_dict()
        scheduler_dict = self.scheduler.state_dict()

        state_dict = {'model': model_dict,
                      'optimizer': optimizer_dict,
                      'scheduler': scheduler_dict,
                      'global_step': self.global_step}

        torch.save(state_dict, path_)

        logger.info(f'State dict was saved to {path_}.')

    def load_state_dict(self, path_):
        if not os.path.exists(path_):
            logger.warning(f'Checkpoint {path_} does not exist, so checkpoint was not loaded.')
            return

        state_dict = torch.load(path_, map_location=self.device)

        model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

        model.load_state_dict(state_dict['model'])
        self.global_step = state_dict['global_step']

        logger.info(f'Model weights were loaded from {path_} checkpoint.')

        if not self.drop_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])

            logger.info(f'Optimizer and scheduler also were restored from {path_} checkpoint.')
