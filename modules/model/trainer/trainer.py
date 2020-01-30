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
from transformers import get_linear_schedule_with_warmup

from .callback import TestCallback
from .meters import *

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


@dataclass
class Trainer:
    model: nn.Module
    loss: Any
    collate_fun: Any

    optimizer: Any = None

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

    warmup_coef: float = 0.01
    max_grad_norm: float = 1

    apex_level: str = None
    apex_verbosity: int = 1
    apex_loss_scale: float = None

    train_weights: defaultdict = None

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

        self.train_dataloader = Trainer._init_dataloader(self.train_dataset,
                                                         'Train',
                                                         batch_size=int(self.train_batch_size // self.batch_split),
                                                         n_jobs=self.n_jobs,
                                                         sampler=self._init_train_sampler(),
                                                         drop_last=True,
                                                         collate_fun=self.collate_fun)

        self.test_dataloader = Trainer._init_dataloader(self.test_dataset,
                                                        'Test',
                                                        batch_size=self.test_batch_size,
                                                        n_jobs=self.n_jobs,
                                                        sampler=None,
                                                        drop_last=False,
                                                        collate_fun=self.collate_fun)

        self.scheduler = None
        use_scheduler = self.train_dataloader is not None and self.optimizer is not None and self.warmup_coef > 0
        if use_scheduler:
            num_training_steps = self.n_epochs * len(self.train_dataloader) // self.batch_split
            num_warmup_steps = int(num_training_steps * self.warmup_coef)

            logger.info(f'Wurmup scheldure is used. #Training steps: {num_training_steps}. '
                        f'#Warmup steps: {num_warmup_steps}.')

            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps,
                                                             num_training_steps=num_training_steps)

        # init apex
        self.model, self.optimizer = initialize_apex(self.model, optimizer=self.optimizer,
                                                     apex_level=self.apex_level,
                                                     apex_verbosity=self.apex_verbosity,
                                                     apex_loss_scale=self.apex_loss_scale)
        logger.info(f'APEX optimization level: {self.apex_level}. APEX verbosity: {self.apex_verbosity}.')

        # init distributed training
        if self.local_rank != -1:
            if self.gpu_id is not None:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.gpu_id], output_device=self.gpu_id, find_unused_parameters=True
                )
            else:
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)

        self.global_step = 0
        self.writer = Trainer._init_writer(self.local_rank, self.writer_dir)

        if self.debug:
            self.n_epochs = 2

    def _init_train_sampler(self):
        if self.train_dataset is None:
            return None

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

        return train_sampler

    @staticmethod
    def _init_dataloader(dataset, name, *, batch_size=1, n_jobs=0, sampler=None, drop_last=False, collate_fun=None):
        if dataset is None:
            return None

        logger.info(f'{name} dataset len: {len(dataset)}. #JOBS: {n_jobs}.')

        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           num_workers=n_jobs,
                                           sampler=sampler,
                                           drop_last=drop_last,
                                           shuffle=False,
                                           collate_fn=collate_fun)

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
        if self.apex_level is None and hasattr(self.model, 'modules'):
            for module in self.model.modules:
                module.train()
        else:
            self.model.train()

    def set_eval(self):
        if self.apex_level is None and hasattr(self.model, 'modules'):
            for module in self.model.modules:
                module.eval()
        else:
            self.model.eval()

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
        if self.train_dataloader is None:
            logger.warning('You have not specified train dataset, so you cannot run train method.')
            return

        after_epoch_funcs = [] if after_epoch_funcs is None else after_epoch_funcs

        def run_after_funcs():
            for func in after_epoch_funcs:
                func(epoch_i)

        for epoch_i in range(1, self.n_epochs+1):
            self._train(epoch_i)
            run_after_funcs()

    def _train(self, epoch_i):
        self.set_train()
        self.optimizer.zero_grad()

        avg_meters = defaultdict(AverageMeter)

        tqdm_data = tqdm(self.train_dataloader, desc=f'Train (epoch #{epoch_i} / {self.n_epochs})')

        for i, (inputs, labels) in enumerate(tqdm_data):
            inputs, labels = self._to_device((inputs, labels))

            pred_logits = self.model(**inputs)

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
        if self.local_rank in [0, -1]:
            if self.test_dataloader is None:
                logger.warning('You have not specified test dataset, so you cannot run test method.')
                return

            if callbacks is not None and not isinstance(callbacks, (list, tuple)):
                callbacks = tuple(callbacks)

            assert all(isinstance(c, TestCallback) for c in callbacks)

            if self.test_dataloader is not None:
                with torch.no_grad():
                    self._test(epoch_i, callbacks=callbacks)

        if self.local_rank != -1:
            logger.warning('Waiting till validation ends in main process..')
            torch.distributed.barrier()

    @torch.no_grad()
    def _test(self, epoch_i, *, callbacks=None):
        self.set_eval()

        avg_meters = defaultdict(AverageMeter)
        tqdm_data = tqdm(self.test_dataloader, desc=f'Test (epoch #{epoch_i} / {self.n_epochs})')

        for i, (inputs, labels) in enumerate(tqdm_data):
            inputs, labels = self._to_device((inputs, labels))

            pred_logits = self.model(**inputs)

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
