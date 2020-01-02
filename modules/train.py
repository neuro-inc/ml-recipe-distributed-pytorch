import logging
import os
import random
from pathlib import Path

import configargparse
import numpy as np
import torch
import torch.multiprocessing as mp
from model.dataset_online import DataPreprocessorOnline, DataPreprocessorDatasetOnline
from model.model import BertForQuestionAnswering
from model.trainer import Trainer
from transformers import BertTokenizer


def get_parser() -> configargparse.ArgumentParser:
    def cast2(type_):
        return lambda x: type_(x) if x != 'None' else None

    parser = configargparse.ArgumentParser(description='Midi-generator training script.')

    parser.add_argument('-c', '--config_file', required=False, is_config_file=True, help='Config file path.')

    parser.add_argument('--dump_dir', type=Path, default='../results', help='Dump path.')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name.')

    parser.add_argument('--last', type=cast2(str), default=None, help='Restored checkpoint.')

    parser.add_argument('--gpu', action='store_true', help='Use gpu to train model.')

    parser.add_argument('--seed', type=cast2(int), default=None, help='Seed for random state.')

    parser.add_argument('--n_jobs', type=int, default=2, help='Number of threads in data loader.')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs.')

    parser.add_argument('--train_batch_size', type=int, default=128, help='Number of items in batch.')
    parser.add_argument('--test_batch_size', type=int, default=16, help='Number of items in batch.')
    parser.add_argument('--batch_split', type=int, default=1,
                        help='Batch will be split into this number chunks during training.')

    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer.')

    parser.add_argument('--data_path', type=str, required=True, help='')
    parser.add_argument('--label_info_dump', type=str, default='./data/train_labels.pkl', help='')
    parser.add_argument('--split_info_dump', type=str, default='./data/train_split.pkl', help='')

    parser.add_argument('--w_start', type=float, default=1, help='')
    parser.add_argument('--w_end', type=float, default=1, help='')
    parser.add_argument('--w_cls', type=float, default=1, help='')

    parser.add_argument('--max_grad_norm', type=float, default=1, help='')

    parser.add_argument('--warmup_coef', type=float, default=0.01, help='')

    parser.add_argument('--apex_level', type=cast2(str), default=None, help='')
    parser.add_argument('--apex_verbosity', type=int, default=1, help='')

    parser.add_argument('--drop_optimizer', action='store_true', help='')

    parser.add_argument('--debug', action='store_true', help='Debug mode.')

    parser.add_argument('--local_rank', type=int, default=-1, help='')

    parser.add_argument('--dist_backend', type=str, default='nccl', help='')
    parser.add_argument('--dist_init_method', type=str, default='tcp://127.0.0.1:9080', help='')
    parser.add_argument('--dist_world_size', type=int, default=1, help='')

    return parser


def set_seed(seed=None):
    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(seed)

        random.seed(seed)
        np.random.seed(seed)

        logger.info(f'Random seed was set to {seed}. It can affect speed of training.')


def get_model(params):
    bert_model = 'bert-base-uncased'
    do_lower_case = 'uncased' in bert_model

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    model = BertForQuestionAnswering.from_pretrained(bert_model, num_labels=len(DataPreprocessorOnline.labels2id))

    return model, tokenizer


def get_datasets(params, tokenizer):
    preprocessor = DataPreprocessorOnline(params.data_path,
                                          label_info_dump=params.label_info_dump,
                                          split_info_dump=params.split_info_dump)

    label_counter, labels = preprocessor.scan_labels()
    train_indexes, train_labels, test_indexes, test_labels = preprocessor.split_train_test(labels)

    train_weights = np.asarray([1 / (label_counter[label]) for label in train_labels])
    train_weights = train_weights / np.sum(train_weights)

    train_dataset = DataPreprocessorDatasetOnline(preprocessor, tokenizer, train_indexes)
    test_dataset = DataPreprocessorDatasetOnline(preprocessor, tokenizer, test_indexes)

    return train_dataset, test_dataset, train_weights


def show_params(params):
    logger.info('Input parameters:')
    for k in sorted(params.__dict__.keys()):
        logger.info(f'\t{k}: {getattr(params, k)}')


def run_worker(device, params):
    if params.distributed:
        if params.local_rank == -1:
            raise AttributeError('Specify local rank.')

        if params.dist_multiprocessing:
            params.local_rank = params.local_rank * params.dist_ngpus_per_node + device

        torch.distributed.init_process_group(backend=params.dist_backend, init_method=params.dist_init_method,
                                             world_size=params.dist_world_size, rank=params.local_rank)

        # todo: incorrect without multiprocessing?
        torch.cuda.set_device(device)
        device = torch.device('cuda', params.local_rank)

    logger = get_logger(level=logging.INFO if params.local_rank in [-1, 0] else logging.WARN)
    logger.info(f'Used device in main process: {device}')

    model, tokenizer = get_model(params)
    train_dataset, test_dataset, train_weights = get_datasets(params, tokenizer)

    trainer = Trainer(model, tokenizer, train_dataset, test_dataset,
                      writer_dir=params.dump_dir / f'board/{params.experiment_name}',
                      device=device,
                      train_batch_size=params.train_batch_size,
                      test_batch_size=params.test_batch_size,
                      batch_split=params.batch_split,
                      n_jobs=params.n_jobs,
                      n_epochs=params.n_epochs,
                      lr=params.lr,
                      weight_decay=params.weight_decay,
                      w_start=params.w_start,
                      w_end=params.w_end,
                      w_cls=params.w_cls,
                      warmup_coef=params.warmup_coef,
                      apex_level=params.apex_level,
                      apex_verbosity=params.apex_verbosity,
                      train_weights=train_weights,
                      drop_optimizer=params.drop_optimizer,
                      max_grad_norm=params.max_grad_norm,
                      debug=params.debug)

    if params.last is not None:
        trainer.load_state_dict(params.last)

    # help functions
    def save_last(*args, **kwargs):
        trainer.save_state_dict(params.dump_dir / params.experiment_name / 'last.ch')

    def save_each(epoch_i):
        trainer.save_state_dict(params.dump_dir / params.experiment_name / f'epoch_{epoch_i}.ch')

    # class save_best:
    #     def __init__(self):
    #         self.metric = trainer_config.best_metric
    #         self.order = trainer_config.best_order
    #         self.value = 0
    #
    #     def __call__(self, *args):
    #         assert hasattr(trainer, 'metrics')
    #         assert self.metric in trainer.metrics
    #
    #         if eval(f'{trainer.metrics[self.metric]}{self.order}{self.value}'):
    #             self.value = trainer.metrics[self.metric]
    #             logger.info(f'Best value of {self.metric} was achieved after training step {trainer.global_step} '
    #                         f'and equals to {self.value}')
    #             state_dict = trainer.state_dict()
    #             torch.save(state_dict, join(trainer_config.dump_dir, 'best.ch'))

    try:
        trainer.train(after_epoch_funcs=[save_last, save_each, trainer.test])
    except KeyboardInterrupt:
        logger.error('Training process was interrupted.')
        trainer.save_state_dict(params.dump_dir / params.experiment_name / 'interrupt.ch')


def main() -> None:
    params = get_parser().parse_args()

    show_params(params)
    os.makedirs(params.dump_dir / params.experiment_name, exist_ok=True)

    set_seed(params.seed)

    # todo: wrong rank if nodes have different gpu number?
    params.dist_ngpus_per_node = torch.cuda.device_count()
    params.dist_world_size *= params.dist_ngpus_per_node
    params.distributed = params.dist_world_size > 1

    logger.info(f'Distributed: {params.distributed}. Distributed multiprocessing: {params.dist_multiprocessing}. '
                f'World size: {params.dist_world_size}, #GPU: {params.dist_ngpus_per_node}.')

    if params.dist_ngpus_per_node > 1:
        mp.spawn(run_worker, nprocs=params.dist_ngpus_per_node, args=(params,))
    else:
        device = torch.device('cuda') if torch.cuda.is_available() and params.gpu else torch.device('cpu')
        run_worker(device, params)


def get_logger(level=logging.INFO):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=level)

    logging.getLogger('transformers').setLevel('CRITICAL')

    return logging.getLogger(__file__)


if __name__ == '__main__':
    logger = get_logger()

    main()
