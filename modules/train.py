import logging
import os
import random
from pathlib import Path

import configargparse
import numpy as np
import torch
from model.dataset_online import DataPreprocessorOnline, DataPreprocessorDatasetOnline
from model.model import BertForQuestionAnswering
from model.trainer import Trainer
from transformers import BertTokenizer


def set_seed(logger, seed=None):
    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(seed)

        random.seed(seed)
        np.random.seed(seed)

        logger.info(f'Random seed was set to {seed}.')


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
    parser.add_argument('--dist_init_method', type=str, default='env://', help='')
    parser.add_argument('--dist_world_size', type=int, default=1, help='')

    return parser


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


def show_params(params, logger):
    logger.info('Input parameters:')
    for k in sorted(params.__dict__.keys()):
        logger.info(f'\t{k}: {getattr(params, k)}')


def run_worker(device, params):
    pass


def main() -> None:
    params = get_parser().parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if params.local_rank in [-1, 0] else logging.WARN)

    logging.getLogger('transformers').setLevel('CRITICAL')
    logger = logging.getLogger(__file__)

    show_params(params, logger)
    os.makedirs(params.dump_dir / params.experiment_name, exist_ok=True)

    set_seed(logger, params.seed)

    if params.local_rank == -1 and params.gpu:
        device = torch.device('cuda') if torch.cuda.is_available() and params.gpu else torch.device('cpu')
    else:
        # torch.cuda.set_device(params.local_rank)
        device = torch.device('cuda') #, params.local_rank)
        torch.distributed.init_process_group(backend=params.dist_backend, init_method=params.dist_init_method,
                                             world_size=params.dist_world_size, rank=params.local_rank)

    bert_model = 'bert-base-uncased'
    do_lower_case = 'uncased' in bert_model

    # if params.local_rank == 0:
    #     torch.distributed.barrier()

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    model = BertForQuestionAnswering.from_pretrained(bert_model, num_labels=len(DataPreprocessorOnline.labels2id))

    # if params.local_rank == 0:
    #     torch.distributed.barrier()

    train_dataset, test_dataset, train_weights = get_datasets(params, tokenizer)

    trainer = Trainer(model, tokenizer, train_dataset, test_dataset,
                      # writer=writer,
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


if __name__ == '__main__':
    main()
