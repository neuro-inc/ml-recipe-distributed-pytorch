import logging
import os

import torch
import torch.nn as nn
# from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from .loss import FocalLossWithLogits
from .meters import *
from .split_dataset import RawPreprocessor

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


def get_optimized_parameters(model, weight_decay, *,
                             finetune=False,
                             finetune_transformer=False,
                             finetune_position=False,
                             finetune_class=False):
    if finetune:
        model.eval()

        optimizer_parameters = []
        modules = []

        if finetune_transformer:
            modules.append(model.transformer)
            optimizer_parameters.extend(list(modules[-1].named_parameters()))

        if finetune_position:
            modules.append(model.position_outputs)
            optimizer_parameters.extend(list(modules[-1].named_parameters()))

        if finetune_class:
            modules.append(model.classifier)
            optimizer_parameters.extend(list(modules[-1].named_parameters()))

        if not modules:
            raise AttributeError('Specify at least one module for fine-tuning.')

        logger.info(f'Fine-tuned modules: transformer({finetune_transformer}), '
                    f'position({finetune_position}),  classifier({finetune_class}).')

    else:
        modules = [model]
        optimizer_parameters = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in optimizer_parameters if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in optimizer_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return modules, optimizer_grouped_parameters


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
                 focal=False,
                 focal_alpha=1,
                 focal_gamma=2,
                 warmup_coef=0.01,
                 apex_level=None,
                 apex_verbosity=1,
                 drop_optimizer=False,
                 train_weights=None,
                 debug=False,
                 max_grad_norm=1,
                 local_rank=-1,
                 gpu_id=None,
                 sync_bn=False,
                 finetune=False,
                 finetune_transformer=False,
                 finetune_position=False,
                 finetune_class=False
                 ):

        if sync_bn and local_rank != -1:
            try:
                import apex
                model = apex.parallel.convert_syncbn_model(model)
                logger.info('BatchNorm was synchronized across nodes with APEX module.')
            except ImportError:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                logger.info('BatchNorm was synchronized across nodes with Pytorch module.')

        self.model = model.to(device)

        self.modules, optimizer_grouped_parameters = get_optimized_parameters(self.model, weight_decay,
                                                                              finetune=finetune,
                                                                              finetune_transformer=finetune_transformer,
                                                                              finetune_position=finetune_position,
                                                                              finetune_class=finetune_class)

        # todo: incorrect value during distributed training
        num_training_steps = n_epochs * len(train_dataset) // train_batch_size
        num_warmup_steps = int(num_training_steps * warmup_coef)

        logger.info(f'Train Dataset len: {len(train_dataset)}. #JOBS: {n_jobs}.')
        logger.info(f'Test Dataset len: {len(test_dataset)}. #JOBS: {n_jobs}.')
        logger.info(f'Training steps number: {num_training_steps}. Warmup steps number: {num_warmup_steps}.')

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps,
                                                         num_training_steps=num_training_steps) \
            if num_warmup_steps else None

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

        if local_rank == -1:
            if train_weights is None or train_weights['sampler_weights'] is None:
                train_sampler = RandomSampler(train_dataset)
            else:
                assert len(train_weights['sampler_weights']) == len(train_dataset)
                train_sampler = WeightedRandomSampler(train_weights['sampler_weights'],
                                                      len(train_dataset))
        else:
            train_sampler = DistributedSampler(train_dataset)

        logger.info(f'Used train sampler: {type(train_sampler).__name__}.')

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=int(train_batch_size // batch_split),
                                                            num_workers=n_jobs,
                                                            sampler=train_sampler,
                                                            drop_last=True,
                                                            collate_fn=self.collate_fun) \
            if train_dataset is not None else None

        self.test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                           batch_size=test_batch_size,
                                                           num_workers=n_jobs,
                                                           shuffle=False,
                                                           drop_last=False,
                                                           collate_fn=self.collate_fun) \
            if test_dataset is not None else None

        self.start_loss = FocalLossWithLogits(alpha=focal_alpha, gamma=focal_gamma, ignore_index=-1) if focal \
            else nn.CrossEntropyLoss(ignore_index=-1)
        self.end_loss = FocalLossWithLogits(alpha=focal_alpha, gamma=focal_gamma, ignore_index=-1) if focal \
            else nn.CrossEntropyLoss(ignore_index=-1)
        self.cls_loss = FocalLossWithLogits(alpha=focal_alpha, gamma=focal_gamma) if focal \
            else nn.CrossEntropyLoss(weight=self._to_device(train_weights['label_weights']))

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
        self.drop_optimizer = drop_optimizer

        self.debug = debug
        if self.debug:
            self.n_epochs = 1

        self.writer = SummaryWriter(log_dir=writer_dir) if self.local_rank in [-1, 0] else None

        self.metrics = None

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def collate_fun(self, items):
        batch_size = len(items)
        pad_token_id = self.tokenizer.pad_token_id

        max_len = max([len(item.input_ids) for item in items])
        tokens = pad_token_id * np.ones((batch_size, max_len), dtype=np.int64)
        # todo: wtf
        token_type_ids = np.zeros((batch_size, max_len), dtype=np.int64) if self.tokenizer.model_name == 'roberta' \
            else np.ones((batch_size, max_len), dtype=np.int64)

        for i, item in enumerate(items):
            row = item.input_ids

            tokens[i, :len(row)] = row
            # todo: wtf
            if self.tokenizer.model_name == 'bert':
                token_type_id = [0 if i <= row.index(self.tokenizer.sep_token_id) else 1 for i in range(len(row))]
                token_type_ids[i, :len(row)] = token_type_id

        attention_mask = tokens > 0
        inputs = [torch.from_numpy(tokens),
                  torch.from_numpy(attention_mask),
                  torch.from_numpy(token_type_ids),
                  ]

        # output labels
        start_ids = np.array([item.start_id for item in items])
        end_ids = np.array([item.end_id for item in items])

        label_ids = [item.label_id for item in items]

        labels = [torch.LongTensor(start_ids),
                  torch.LongTensor(end_ids),
                  torch.LongTensor(label_ids),
                  ]

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

    def set_train(self):
        if self.apex_level is not None:
            self.model.train()
        else:
            for module in self.modules:
                module.train()

    def set_eval(self):
        if self.apex_level is not None:
            self.model.eval()
        else:
            for module in self.modules:
                module.eval()

    def _to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [self._to_device(d) for d in data]
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif data is None:
            return data
        else:
            raise NotImplemented

    def train(self, after_epoch_funcs=None):
        after_epoch_funcs = [] if after_epoch_funcs is None else after_epoch_funcs

        def run_after_funcs():
            for func in after_epoch_funcs:
                func(epoch_i)

        for epoch_i in range(1, self.n_epochs+1):
            if self.train_dataloader is not None:
                self._train(epoch_i)
            run_after_funcs()

    def _train(self, epoch_i):
        self.set_train()
        self.optimizer.zero_grad()

        avg_losses = defaultdict(AverageMeter)

        tqdm_data = tqdm(self.train_dataloader, desc=f'Train (epoch #{epoch_i} / {self.n_epochs})')

        for i, (inputs, labels) in enumerate(tqdm_data):
            (input_ids, attention_mask, token_type_ids), labels = self._to_device((inputs, labels))

            pred_logits = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)

            self._backward(self._loss(pred_logits, labels, avg_losses=avg_losses))

            avg_losses['lr'] = self.get_lr()

            if (i + 1) % self.batch_split == 0:
                self._clip_grad_norm()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self._update_writer(avg_losses, prefix='train')

                self.global_step += 1

                if self.debug:
                    logger.info('Training was interrupted because of debug mode.')
                    break

            Trainer._update_console(tqdm_data, avg_losses)

    def test(self, epoch_i):
        if self.test_dataloader is not None:
            with torch.no_grad():
                self._test(epoch_i)

        if self.local_rank != -1:
            # Wait till validation ends in main process
            torch.distributed.barrier()

    def _test(self, epoch_i):
        self.set_eval()

        avg_losses = defaultdict(AverageMeter)
        map_meter = MAPMeter()

        tqdm_data = tqdm(self.test_dataloader, desc=f'Test (epoch #{epoch_i} / {self.n_epochs})')

        for i, (inputs, labels) in enumerate(tqdm_data):
            (input_ids, attention_mask, token_type_ids), labels = self._to_device((inputs, labels))

            pred_logits = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)

            _ = self._loss(pred_logits, labels, avg_losses=avg_losses)

            start_logits, end_logits, cls_logits = (logits.detach().cpu() for logits in pred_logits)
            start_true, end_true, cls_true = (label.detach().cpu() for label in labels)

            start_pred, end_pred, cls_pred = (torch.max(logits, dim=-1)[1] for logits in (start_logits, end_logits, cls_logits))

            start_idxs = start_true != -1
            end_idxs = end_true != -1

            if any(start_idxs):
                avg_losses['s_acc'].update(metrics.accuracy_score(start_true[start_idxs], start_pred[start_idxs]))
            if any(end_idxs):
                avg_losses['e_acc'].update(metrics.accuracy_score(end_true[end_idxs], end_pred[end_idxs]))
            avg_losses['c_acc'].update(metrics.accuracy_score(cls_true, cls_pred))

            map_meter.update(keys=list(RawPreprocessor.labels2id.keys()),
                             pred_probas=torch.softmax(cls_logits, dim=-1).numpy(),
                             true_labels=cls_true.numpy())

            Trainer._update_console(tqdm_data, avg_losses)

            if self.debug:
                logger.info('Test was interrupted because of debug mode.')
                break

        avg_losses.update(map_meter())
        self._update_writer(avg_losses, prefix='test')

        out_dict = {k: v() if isinstance(v, AverageMeter) else v for k, v in avg_losses.items()}
        logger.info(f'Test metrics after epoch {epoch_i}: {out_dict}')
        self.metrics = out_dict

    def save_state_dict(self, path_):
        if self.local_rank not in [-1, 0]:
            return

        if self.debug:
            logger.info(f'Model was not saved to {path_} because of debug mode.')
            return

        model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

        model_dict = model.state_dict()
        optimizer_dict = self.optimizer.state_dict()
        scheduler_dict = self.scheduler.state_dict() if self.scheduler is not None else None

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
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state_dict['scheduler'])

            logger.info(f'Optimizer and scheduler also were restored from {path_} checkpoint.')
