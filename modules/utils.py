import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_logger(*, level=logging.INFO, filename=None, filemode='w', logger_name=None, debug=False):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handlers = [logging.StreamHandler()]
    if filename is not None:
        handlers.append(logging.FileHandler(filename, filemode))

    path_format = '%(pathname)s:%(funcName)s:%(lineno)d' if debug else '%(name)s'

    logging.basicConfig(format=f'%(asctime)s - %(levelname)s - {path_format} -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=level,
                        handlers=handlers)

    logging.getLogger('transformers').setLevel('CRITICAL')

    logger = logging.getLogger(__file__ if logger_name is None else logger_name)
    if filename is not None and filemode == 'w':
        logger.info(f'All logs will be dumped to {filename}.')

    return logger


def set_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.info(f'Random seed was set to {seed}. It can affect speed of training and performance of result model.')


def show_params(params, name):
    logger.info(f'Input {name} parameters:')
    for k in sorted(params.__dict__.keys()):
        logger.info(f'\t\t{k}: {getattr(params, k)}')
