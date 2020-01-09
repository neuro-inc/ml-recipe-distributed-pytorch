import math
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp
from model.model import BertForQuestionAnswering
from model.parser import get_trainer_parser, get_model_parser, write_config_file
from model.split_dataset import RawPreprocessor, SplitDataset
from model.trainer import Trainer
from transformers import BertTokenizer


def set_seed(seed=None):
    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(seed)

        random.seed(seed)
        np.random.seed(seed)

        logger.info(f'Random seed was set to {seed}. It can affect speed of training and performance of result model.')


def get_model(model_params):
    bert_model = model_params.model
    do_lower_case = 'uncased' in bert_model

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    model = BertForQuestionAnswering.from_pretrained(bert_model, num_labels=len(RawPreprocessor.labels2id))

    return model, tokenizer


def get_datasets(params, *, tokenizer=None, clear=False):
    preprocessor = RawPreprocessor(raw_json=params.data_path,
                                   out_dir=params.processed_data_path,
                                   clear=clear)
    labels_counter, labels, (train_indexes, train_labels, test_indexes, test_labels) = preprocessor()

    train_weights = np.asarray([1 / (labels_counter[label]) for label in train_labels])
    train_weights = train_weights / np.sum(train_weights)

    train_dataset = SplitDataset(params.processed_data_path, tokenizer, train_indexes,
                                 max_seq_len=params.max_seq_len,
                                 max_question_len=params.max_question_len,
                                 doc_stride=params.doc_stride)
    test_dataset = SplitDataset(params.processed_data_path, tokenizer, test_indexes, test=True,
                                max_seq_len=params.max_seq_len,
                                max_question_len=params.max_question_len,
                                doc_stride=params.doc_stride) \
        if params.local_rank in [-1, 0] else None

    return train_dataset, test_dataset, train_weights


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
    logger = get_logger(level=log_level, filename=log_file, filemode='a')

    logger.warning(f'Process with local_rank: {params.local_rank}. Used device: {device}. GPU id: {gpu_id}.')

    model, tokenizer = get_model(model_params)
    train_dataset, test_dataset, train_weights = get_datasets(params, tokenizer=tokenizer, clear=False)

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
                      focal=params.focal,
                      focal_alpha=params.focal_alpha,
                      focal_gamma=params.focal_gamma,
                      warmup_coef=params.warmup_coef,
                      apex_level=params.apex_level,
                      apex_verbosity=params.apex_verbosity,
                      train_weights=train_weights,
                      drop_optimizer=params.drop_optimizer,
                      max_grad_norm=params.max_grad_norm,
                      sync_bn=params.sync_bn,
                      debug=params.debug,
                      local_rank=params.local_rank,
                      gpu_id=gpu_id,
                      finetune=params.finetune,
                      finetune_transformer=params.finetune_transformer,
                      finetune_position=params.finetune_position,
                      finetune_class=params.finetune_class
                      )

    if params.last is not None:
        trainer.load_state_dict(params.last)

    # helpers
    def save_last(*args, **kwargs):
        trainer.save_state_dict(params.dump_dir / params.experiment_name / 'last.ch')

    def save_each(epoch_i):
        trainer.save_state_dict(params.dump_dir / params.experiment_name / f'epoch_{epoch_i}.ch')

    class save_best:
        def __init__(self):
            self.metric = params.best_metric
            self.order = params.best_order
            self.value = 1e10 * (-1 if params.best_order == '>' else 1)

        def __call__(self, *args):
            if trainer.metrics is not None:
                assert self.metric in trainer.metrics

                if self.metric in trainer.metrics and not math.isnan(trainer.metrics[self.metric]):

                    if eval(f'{trainer.metrics[self.metric]}{self.order}{self.value}'):
                        self.value = trainer.metrics[self.metric]
                        trainer.save_state_dict(params.dump_dir / params.experiment_name / f'best.ch')
                        logger.info(f'Best value of {self.metric} was achieved after training step {trainer.global_step} '
                                    f'and equals to {self.value}')
                else:
                    logger.warning(f'Trainer metrics do not contain metric {self.metric}.')

    try:
        trainer.train(after_epoch_funcs=[save_last, save_each, trainer.test, save_best()])
    except KeyboardInterrupt:
        logger.error('Training process was interrupted.')
        trainer.save_state_dict(params.dump_dir / params.experiment_name / 'interrupt.ch')
    except Exception as e:
        logger.error(e)
        raise e


def show_params(params, name):
    logger.info(f'Input {name} parameters:')
    for k in sorted(params.__dict__.keys()):
        logger.info(f'\t{k}: {getattr(params, k)}')


def main(params, model_params) -> None:
    show_params(model_params, 'model')
    show_params(params, 'trainer')

    set_seed(params.seed)

    # todo: wrong rank if nodes have different gpu number or when node does not gpu
    params.dist_ngpus_per_node = torch.cuda.device_count()
    params.dist_world_size *= params.dist_ngpus_per_node
    params.distributed = params.dist_world_size > 1
    params.distributed_mp = params.dist_ngpus_per_node > 1

    logger.info(f'Distributed: {params.distributed}. Distributed multiprocessing: {params.distributed_mp}. '
                f'World size: {params.dist_world_size}, #GPU: {params.dist_ngpus_per_node}.')

    if params.distributed and params.local_rank in [0, -1]:
        logger.info('Preparing dataset in main process.')
        _ = get_datasets(params, tokenizer=None, clear=params.clear_processed)

    if params.distributed_mp:
        mp.spawn(run_worker, nprocs=params.dist_ngpus_per_node, args=(params, model_params))
    else:
        device = torch.device('cuda') if torch.cuda.is_available() and params.gpu else torch.device('cpu')
        run_worker(device, params, model_params)


def get_logger(level=logging.INFO, filename=None, filemode='w'):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=level,
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(filename, filemode)])

    logging.getLogger('transformers').setLevel('CRITICAL')

    logger = logging.getLogger(__file__)
    if filename is not None and filemode == 'w':
        logger.info(f'All logs will be dumped to {filename}.')

    return logger


if __name__ == '__main__':
    parser = get_trainer_parser()
    params, unused = parser.parse_known_args()

    os.makedirs(params.dump_dir / params.experiment_name, exist_ok=True)

    model_parser = get_model_parser()
    model_params, model_unused = model_parser.parse_known_args()

    unused = set(model_unused) & set(unused)
    if unused:
        parser.print_help()
        model_parser.print_help()
        print(f'Incorrect command line parameters: {unused}.')
        exit()

    params.log_file = params.dump_dir / params.experiment_name / f'{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.log' \
        if params.local_rank in [-1, 0] else None

    logger = get_logger(filename=params.log_file, filemode='w')

    if params.local_rank in [0, -1]:
        write_config_file(parser, params, params.dump_dir / params.experiment_name / 'trainer.cfg')
        write_config_file(model_parser, model_params, params.dump_dir / params.experiment_name / 'model.cfg')

    params.n_jobs = min(params.n_jobs, mp.cpu_count() // 2)

    main(params, model_params)
