import functools
import logging
import os
from datetime import datetime

import torch
import torch.multiprocessing as mp

from init import init_loss, init_model, init_datasets, init_collate_fun
from utils import *

from model.callback import MAPCallback, AccuracyCallback, SaveBestCallback
from model.parser import get_trainer_parser, get_model_parser, write_config_file, get_params
from model.trainer import Trainer


def run_worker(device, params, model_params):
    gpu_id = device if params.distributed_mp else None
    if params.distributed:
        if params.local_rank == -1:
            raise AttributeError('Specify local rank.')

        if params.distributed_mp:
            params.local_rank = params.local_rank * params.dist_ngpus_per_node + device

        torch.distributed.init_process_group(backend=params.dist_backend, init_method=params.dist_init_method,
                                             world_size=params.dist_world_size, rank=params.local_rank)

        if params.distributed_mp:
            torch.cuda.set_device(device)
            device = torch.device('cuda', params.local_rank)

            if params.dist_ngpus_per_node * params.n_jobs > mp.cpu_count():
                params.n_jobs = mp.cpu_count() // (2 * params.dist_ngpus_per_node)

        # Wait dataset initialization in main process. Dataset directory mast be shared
        torch.distributed.barrier()

    log_file = params.log_file if params.local_rank in [-1, 0] else None
    log_level = logging.INFO if params.local_rank in [-1, 0] else logging.WARN
    logger = get_logger(level=log_level, filename=log_file, filemode='a', logger_name='train')

    logger.warning(f'Process with local_rank: {params.local_rank}. Used device: {device}. GPU id: {gpu_id}.')

    model, tokenizer = init_model(model_params, bpe_dropout=params.bpe_dropout)
    train_dataset, test_dataset, train_weights = init_datasets(params, tokenizer=tokenizer, clear=False)
    loss = init_loss(params, train_weights)

    trainer = Trainer(model=model,
                      loss=loss,
                      collate_fun=init_collate_fun(tokenizer),

                      train_dataset=train_dataset,
                      test_dataset=test_dataset,

                      writer_dir=params.dump_dir / f'board/{params.experiment_name}',

                      device=device,

                      local_rank=params.local_rank,
                      gpu_id=gpu_id,
                      sync_bn=params.sync_bn,

                      n_epochs=params.n_epochs,

                      train_batch_size=params.train_batch_size,
                      test_batch_size=params.test_batch_size,

                      batch_split=params.batch_split,
                      n_jobs=params.n_jobs,

                      optimizer=params.optimizer,

                      lr=params.lr,
                      weight_decay=params.weight_decay,
                      warmup_coef=params.warmup_coef,
                      max_grad_norm=params.max_grad_norm,

                      apex_level=params.apex_level,
                      apex_verbosity=params.apex_verbosity,
                      apex_loss_scale=params.apex_loss_scale,

                      train_weights=train_weights,

                      finetune=params.finetune,
                      finetune_transformer=params.finetune_transformer,
                      finetune_position=params.finetune_position,
                      finetune_position_reg=params.finetune_position_reg,
                      finetune_class=params.finetune_class,

                      drop_optimizer=params.drop_optimizer,
                      debug=params.debug
                      )

    if params.last is not None:
        trainer.load_state_dict(params.last)

    # helpers
    def save_last(*args, **kwargs):
        trainer.save_state_dict(params.dump_dir / params.experiment_name / 'last.ch')

    def save_each(epoch_i):
        trainer.save_state_dict(params.dump_dir / params.experiment_name / f'epoch_{epoch_i}.ch')

    test_fun = functools.partial(trainer.test, callbacks=[MAPCallback(), AccuracyCallback(), SaveBestCallback(params)])

    try:
        trainer.train(after_epoch_funcs=[save_last, save_each, test_fun])
    except KeyboardInterrupt:
        logger.error('Training process was interrupted.')
        trainer.save_state_dict(params.dump_dir / params.experiment_name / 'interrupt.ch')
    except Exception as e:
        logger.error(e)
        raise e


def main(params, model_params) -> None:
    show_params(model_params, 'model')
    show_params(params, 'trainer')

    set_seed(params.seed)

    # todo: wrong rank if nodes have different gpu number or when node does not have a gpu
    params.dist_ngpus_per_node = torch.cuda.device_count()
    params.dist_world_size *= params.dist_ngpus_per_node
    params.distributed = params.dist_world_size > 1
    params.distributed_mp = params.dist_ngpus_per_node > 1

    logger.info(f'Distributed: {params.distributed}. Distributed multiprocessing: {params.distributed_mp}. '
                f'World size: {params.dist_world_size}, #GPU: {params.dist_ngpus_per_node}.')

    if params.distributed and params.local_rank in [0, -1]:
        logger.info('Preparing dataset in main process.')
        _ = init_datasets(params, tokenizer=None, clear=params.clear_processed)

    if params.distributed_mp:
        mp.spawn(run_worker, nprocs=params.dist_ngpus_per_node, args=(params, model_params))
    else:
        device = torch.device('cuda') if torch.cuda.is_available() and params.gpu else torch.device('cpu')
        run_worker(device, params, model_params)


if __name__ == '__main__':
    (parser, model_parser), (params, model_params) = get_params((get_trainer_parser, get_model_parser))

    os.makedirs(params.dump_dir / params.experiment_name, exist_ok=True)

    params.log_file = params.dump_dir / params.experiment_name / f'{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.log' \
        if params.local_rank in [-1, 0] else None

    params.n_jobs = min(params.n_jobs, mp.cpu_count() // 2)

    logger = get_logger(filename=params.log_file, filemode='w', logger_name='train')

    if params.local_rank in [0, -1]:
        write_config_file(parser, params, params.dump_dir / params.experiment_name / 'trainer.cfg')
        write_config_file(model_parser, model_params, params.dump_dir / params.experiment_name / 'model.cfg')

    main(params, model_params)
