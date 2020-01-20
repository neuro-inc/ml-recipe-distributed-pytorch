import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
# from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from .meters import *
from .optim import AdaMod

logger = logging.getLogger(__file__)


def initialize_apex(model, *, optimizer=None, apex_level=None,
                    apex_loss_scale=None, apex_num_losses=1, apex_verbosity=0):
    if apex_level is not None:
        try:
            from apex import amp
        except ImportError:
            raise ImportError('Install Apex to train model with mixed precision.')

        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=apex_level,
                                          loss_scale=apex_loss_scale,
                                          num_losses=apex_num_losses,
                                          verbosity=apex_verbosity)

    return model, optimizer


def get_optimized_parameters(model, weight_decay, *,
                             finetune=False,
                             finetune_transformer=False,
                             finetune_position=False,
                             finetune_class=False):
    if finetune:
        raise NotImplemented

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


@dataclass
class Trainer:
    model: nn.Module
    loss: Any
    collate_fun: Any

    train_dataset: Any = None
    test_dataset: Any = None

    writer_dir: Any = None

    device: Any = torch.device('cuda')

    local_rank: int = -1
    gpu_id: Optional[int] = None
    sync_bn: bool = False

    n_epochs: int = 0

    train_batch_size: int = 32
    test_batch_size: int = 32

    batch_split: int = 1
    n_jobs: int = 4

    optimizer: str = 'adam'

    lr: float = 1e-3
    weight_decay: float = 5e-4
    warmup_coef: float = 0.01
    max_grad_norm: float = 1

    apex_level: str = None
    apex_verbosity: int = 1
    apex_loss_scale: float = None

    train_weights: defaultdict = None

    finetune: bool = False
    finetune_transformer: bool = False
    finetune_position: bool = False
    finetune_class: bool = False

    drop_optimizer: bool = False
    debug: bool = False

    def __post_init__(self):

        if self.sync_bn and self.local_rank != -1:
            try:
                import apex
                self.model = apex.parallel.convert_syncbn_model(self.model)
                logger.info('BatchNorm was synchronized across nodes with APEX module.')
            except ImportError:
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                logger.info('BatchNorm was synchronized across nodes with Pytorch module.')

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

        if self.finetune and self.apex_level is not None:
            logger.warning(f'Finetune mode is not supported with Apex.')
            self.apex_level = None

        self.modules, optimizer_grouped_parameters = get_optimized_parameters(
            self.model, self.weight_decay,
            finetune=self.finetune,
            finetune_transformer=self.finetune_transformer,
            finetune_position=self.finetune_position,
            finetune_class=self.finetune_class)

        # todo: does not work with none dataset
        # todo: incorrect value during distributed training
        num_training_steps = self.n_epochs * len(self.train_dataset) // self.train_batch_size
        num_warmup_steps = int(num_training_steps * self.warmup_coef)

        logger.info(f'Train Dataset len: {len(self.train_dataset)}. #JOBS: {self.n_jobs}.')
        logger.info(f'Test Dataset len: {len(self.test_dataset)}. #JOBS: {self.n_jobs}.')
        logger.info(f'Training steps number: {num_training_steps}. Warmup steps number: {num_warmup_steps}.')

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, correct_bias=False) if self.optimizer == 'adam' \
            else AdaMod(optimizer_grouped_parameters, lr=self.lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps,
                                                         num_training_steps=num_training_steps) \
            if num_warmup_steps else None

        self.model, self.optimizer = initialize_apex(self.model, optimizer=self.optimizer,
                                                     apex_level=self.apex_level,
                                                     apex_verbosity=self.apex_verbosity,
                                                     apex_loss_scale=self.apex_loss_scale)

        if self.local_rank != -1:
            if self.gpu_id is not None:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.gpu_id], output_device=self.gpu_id, find_unused_parameters=True
                )
            else:
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)

        logger.info(f'APEX optimization level: {self.apex_level}. APEX verbosity: {self.apex_verbosity}.')

        if self.local_rank == -1:
            if self.train_weights is None or self.train_weights['sampler_weights'] is None:
                train_sampler = RandomSampler(self.train_dataset)
            else:
                assert len(self.train_weights['sampler_weights']) == len(self.train_dataset)
                train_sampler = WeightedRandomSampler(self.train_weights['sampler_weights'],
                                                      len(self.train_dataset))
        else:
            train_sampler = DistributedSampler(self.train_dataset)

        logger.info(f'Used train sampler: {type(train_sampler).__name__}.')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=int(self.train_batch_size // self.batch_split),
                                                            num_workers=self.n_jobs,
                                                            sampler=train_sampler,
                                                            drop_last=True,
                                                            collate_fn=self.collate_fun) \
            if self.train_dataset is not None else None

        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                           batch_size=self.test_batch_size,
                                                           num_workers=self.n_jobs,
                                                           shuffle=False,
                                                           drop_last=False,
                                                           collate_fn=self.collate_fun) \
            if self.test_dataset is not None else None

        self.global_step = 0
        self.writer = Trainer._init_writer(self.local_rank, self.writer_dir)

        if self.debug:
            self.n_epochs = 2

    @staticmethod
    def _init_writer(local_rank, writer_dir):
        writer = None
        if writer_dir is not None and local_rank in [-1, 0]:
            logger.warning(f'Directory {writer_dir} will be cleaned before SummaryWriter initialization. '
                           f'To prevent missing important information, use different experiment names.')
            shutil.rmtree(writer_dir, ignore_errors=True)
            writer = SummaryWriter(log_dir=writer_dir)

        return writer

    def _get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def _backward(self, loss):
        loss = loss / self.batch_split

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
    def _get_console_str(losses):
        return ', '.join([f'{k}: {losses[k]() if isinstance(losses[k], AverageMeter) else losses[k]:.3e}'
                         for k in losses.keys()])

    @staticmethod
    def _update_console(tqdm_data, losses):
        tqdm_data.set_postfix_str(Trainer._get_console_str(losses))

    def _update_writer(self, losses, *, prefix=None):
        if self.writer is not None:
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
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
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

        avg_meters = defaultdict(AverageMeter)

        tqdm_data = tqdm(self.train_dataloader, desc=f'Train (epoch #{epoch_i} / {self.n_epochs})')

        for i, (inputs, labels) in enumerate(tqdm_data):
            (input_ids, attention_mask, token_type_ids), labels = self._to_device((inputs, labels))

            pred_logits = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)

            self._backward(self.loss(pred_logits, labels, avg_meters=avg_meters))

            avg_meters['lr'] = self._get_lr()

            if (i + 1) % self.batch_split == 0:
                self._clip_grad_norm()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self._update_writer(avg_meters, prefix='train')

                self.global_step += 1

                if self.debug:
                    logger.info('Training was interrupted because of debug mode.')
                    break

            Trainer._update_console(tqdm_data, avg_meters)

    def test(self, epoch_i, *, callbacks=None):
        if callbacks is not None and not isinstance(callbacks, (list, tuple)):
            callbacks = tuple(callbacks)

        if self.test_dataloader is not None:
            with torch.no_grad():
                self._test(epoch_i, callbacks=callbacks)

        if self.local_rank != -1:
            # Wait till validation ends in main process
            torch.distributed.barrier()

    def _test(self, epoch_i, *, callbacks=None):
        self.set_eval()

        avg_meters = defaultdict(AverageMeter)
        tqdm_data = tqdm(self.test_dataloader, desc=f'Test (epoch #{epoch_i} / {self.n_epochs})')

        for i, (inputs, labels) in enumerate(tqdm_data):
            (input_ids, attention_mask, token_type_ids), labels = self._to_device((inputs, labels))

            pred_logits = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)

            self.loss(pred_logits, labels, avg_meters=avg_meters)

            if callbacks is not None:
                for callback in callbacks:
                    callback.at_iteration_end(pred_logits, labels, avg_meters)

            Trainer._update_console(tqdm_data, avg_meters)

            if self.debug and i >= 10:
                logger.info('Test was interrupted because of debug mode.')
                break

        if callbacks is not None:
            for callback in callbacks:
                callback.at_epoch_end(avg_meters, self)

        self._update_writer(avg_meters, prefix='test')

        metrics = {k: v() if isinstance(v, AverageMeter) else v for k, v in avg_meters.items()}
        logger.info(f'Test metrics after epoch {epoch_i} - {Trainer._get_console_str(metrics)}')

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
