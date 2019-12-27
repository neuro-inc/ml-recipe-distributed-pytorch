import logging
import os
import random
from pathlib import Path

import configargparse
import numpy as np
import torch
from model.dataset import QADataloader
from model.model import BertForQuestionAnswering
from model.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from sklearn.model_selection import train_test_split


def set_seed(seed: int = 0) -> None:
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

    parser.add_argument('--seed', type=int, default=0, help='Seed for random state.')

    parser.add_argument('--n_jobs', type=int, default=2, help='Number of threads in data loader.')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs.')

    parser.add_argument('--train_batch_size', type=int, default=128, help='Number of items in batch.')
    parser.add_argument('--test_batch_size', type=int, default=16, help='Number of items in batch.')
    parser.add_argument('--batch_split', type=int, default=1,
                        help='Batch will be split into this number chunks during training.')

    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer.')

    parser.add_argument('--data_path', type=str, required=True, help='')

    parser.add_argument('--max_seq_len', type=int, default=384, help='')
    parser.add_argument('--max_question_len', type=int, default=64, help='')
    parser.add_argument('--doc_stride', type=int, default=128, help='')

    parser.add_argument('--debug', action='store_true', help='Debug mode.')

    return parser


def show_params(params: configargparse.Namespace) -> None:
    logger.info('Input parameters:')
    for k in sorted(params.__dict__.keys()):
        logger.info(f'\t{k}: {getattr(params, k)}')


def main() -> None:
    params = get_parser().parse_args()
    show_params(params)
    os.makedirs(params.dump_dir, exist_ok=True)

    set_seed(params.seed)

    device = torch.device('cuda') if torch.cuda.is_available() and params.gpu else torch.device('cpu')

    bert_model = 'bert-base-uncased'
    do_lower_case = 'uncased' in bert_model

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    train_dataset = QADataloader(params.data_path)
    test_dataset = QADataloader(params.data_path)
    indexes = np.arange(len(test_dataset))

    train_indexes, test_indexes = train_test_split(indexes, test_size=0.15, random_state=params.seed)

    train_dataset.indexes = train_indexes
    test_dataset.indexes = test_indexes

    model = BertForQuestionAnswering.from_pretrained(bert_model, num_labels=len(train_dataset.labels2id))

    writer = SummaryWriter(log_dir=params.dump_dir / f'board/{params.experiment_name}')

    trainer = Trainer(model, tokenizer, train_dataset, test_dataset,
                      writer=writer,
                      device=device,
                      train_batch_size=params.train_batch_size,
                      test_batch_size=params.test_batch_size,
                      batch_split=params.batch_split,
                      n_jobs=params.n_jobs,
                      n_epochs=params.n_epochs,
                      lr=params.lr,
                      weight_decay=params.weight_decay,
                      debug=params.debug)

    if params.last is not None:
        trainer.load_state_dict(params.last)

    # help functions
    def save_last(*args, **kwargs):
        trainer.save_state_dict(params.dump_dir / params.experiment_name / 'last.ch')

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

    trainer.train(after_epoch_funcs=[save_last, trainer.test])
    trainer.save_state_dict(params.dump_dir / f'{params.experiment_name}.ch')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.getLogger('transformers').setLevel('CRITICAL')
    logger = logging.getLogger(__file__)

    main()
