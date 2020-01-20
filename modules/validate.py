import functools
import multiprocessing as mp

import numpy as np
import torch
# from model.model import get_model
from model.parser import get_model_parser, get_predictor_parser, get_params
from model.predictor import Predictor
from model.split_dataset import RawPreprocessor
from model.validation_dataset import ChunkDataset
from utils import *
from init import init_collate_fun, init_model

from transformers import BertTokenizer


def get_validation_dataset(params, *, tokenizer=None, clear=False):
    preprocessor = RawPreprocessor(raw_json=params.data_path,
                                   out_dir=params.processed_data_path,
                                   clear=clear)
    _, _, (_, _, val_indexes, val_labels) = preprocessor()

    val_dataset = ChunkDataset(params.processed_data_path, tokenizer, val_indexes)

    return val_dataset


def main(params, model_params):
    show_params(model_params, 'predictor')
    show_params(params, 'model')

    device = torch.device('cuda') if torch.cuda.is_available() and params.gpu else torch.device('cpu')

    model, tokenizer = init_model(model_params, checkpoint=params.checkpoint, device=device)

    # todo: wtf?
    tokenizer = BertTokenizer.from_pretrained(model_params.model)
    tokenizer.model_name = 'bert'

    val_dataset = get_validation_dataset(params, tokenizer=tokenizer, clear=False)

    collate_fun = init_collate_fun(tokenizer, return_items=True)
    predictor = Predictor(model, device,
                          collate_fun=collate_fun,
                          batch_size=params.batch_size,
                          n_jobs=params.n_jobs,
                          buffer_size=params.buffer_size,
                          limit=params.limit)

    predictor(val_dataset)
    predictor.show_predictions()


if __name__ == '__main__':
    _, (params, model_params) = get_params((get_predictor_parser, get_model_parser))
    logger = get_logger(logger_name='validate')

    params.n_jobs = min(params.n_jobs, mp.cpu_count() // 2)

    main(params, model_params)
