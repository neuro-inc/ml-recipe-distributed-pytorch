import logging
from pathlib import Path

import configargparse

logger = logging.getLogger(__file__)


def cast2(type_):
    return lambda x: type_(x) if x != 'None' else None


def write_config_file(parser, parsed_namespace, output_path):
    config_items = {k: getattr(parsed_namespace, k) for k in sorted(parsed_namespace.__dict__.keys())
                    if 'config' not in k}
    file_contents = parser._config_file_parser.serialize(config_items)

    try:
        with open(output_path, 'w') as output_file:
            output_file.write(file_contents)
    except IOError as e:
        logger.error(f'Could not open file {output_path}.')
        raise e

    logger.info(f'Config was saved to {output_path}.')


def get_model_parser() -> configargparse.ArgumentParser:
    parser = configargparse.ArgumentParser(description='Model config parser.')

    parser.add_argument('-c', '--config_file', required=False, is_config_file=True, help='Config file path.')
    parser.add_argument('--model_config_file', required=False, is_config_file=True, help='Model config file path.')

    parser.add_argument('--model', type=str, default='bert-base-uncased', help='Transformer model name.')

    return parser


def get_trainer_parser() -> configargparse.ArgumentParser:

    parser = configargparse.ArgumentParser(description='Trainer config parser.')

    parser.add_argument('-c', '--config_file', required=False, is_config_file=True, help='Config file path.')
    parser.add_argument('--trainer_config_file', required=False, is_config_file=True, help='Trainer config file path.')

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
    parser.add_argument('--processed_data_path', type=str, required=True, help='')
    parser.add_argument('--clear_processed', action='store_true', help='')

    parser.add_argument('--w_start', type=float, default=1, help='')
    parser.add_argument('--w_end', type=float, default=1, help='')
    parser.add_argument('--w_cls', type=float, default=1, help='')

    parser.add_argument('--focal', action='store_true', help='')
    parser.add_argument('--focal_alpha', type=float, default=1, help='')
    parser.add_argument('--focal_gamma', type=float, default=2, help='')

    parser.add_argument('--max_grad_norm', type=float, default=1, help='')
    parser.add_argument('--sync_bn', action='store_true', help='')

    parser.add_argument('--warmup_coef', type=float, default=0.05, help='')

    parser.add_argument('--apex_level', type=cast2(str), default=None, help='')
    parser.add_argument('--apex_verbosity', type=int, default=1, help='')

    parser.add_argument('--drop_optimizer', action='store_true', help='')

    parser.add_argument('--debug', action='store_true', help='Debug mode.')

    parser.add_argument('--local_rank', type=int, default=-1, help='')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='')
    parser.add_argument('--dist_init_method', type=str, default='tcp://127.0.0.1:9080', help='')
    parser.add_argument('--dist_world_size', type=int, default=1, help='')

    parser.add_argument('--best_metric', choices=['map'], type=str, default='map', help='')
    parser.add_argument('--best_order', choices=['>', '<'], type=str, default='>', help='')

    parser.add_argument('--finetune', action='store_true', help='')
    parser.add_argument('--finetune_transformer', action='store_true', help='')
    parser.add_argument('--finetune_position', action='store_true', help='')
    parser.add_argument('--finetune_class', action='store_true', help='')

    parser.add_argument('--max_seq_len', type=int, default=384, help='')
    parser.add_argument('--max_question_len', type=int, default=64, help='')
    parser.add_argument('--doc_stride', type=int, default=128, help='')

    return parser
