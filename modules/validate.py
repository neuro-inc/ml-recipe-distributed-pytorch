import functools
import multiprocessing as mp

import numpy as np
import torch
from model.model import get_model
from model.parser import get_model_parser, get_predictor_parser, get_params
from model.predictor import Predictor
from model.split_dataset import RawPreprocessor
from model.validation_dataset import ChunkDataset
from utils import *


def collate_fun(items, tokenizer):
    batch_size = len(items)
    pad_token_id = tokenizer.pad_token_id

    max_len = max([len(item.input_ids) for item in items])
    tokens = pad_token_id * np.ones((batch_size, max_len), dtype=np.int64)
    token_type_ids = np.ones((batch_size, max_len), dtype=np.int64)

    for i, item in enumerate(items):
        row = item.input_ids

        tokens[i, :len(row)] = row
        token_type_id = [0 if i <= row.index(102) else 1
                         for i in range(len(row))]  # 102 corresponds to [SEP]

        token_type_ids[i, :len(row)] = token_type_id

    attention_mask = tokens > 0
    inputs = [torch.from_numpy(tokens),
              torch.from_numpy(attention_mask),
              torch.from_numpy(token_type_ids),
              ]

    # output labels
    start_ids = np.array([item.start_id for item in items])
    end_ids = np.array([item.end_id for item in items])

    label_ids = [item.label_id for item in items]

    labels = [torch.LongTensor(start_ids),
              torch.LongTensor(end_ids),
              torch.LongTensor(label_ids),
              ]

    return [inputs, labels, items]


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

    model, tokenizer = get_model(model_params, checkpoint=params.checkpoint, device=device)
    val_dataset = get_validation_dataset(params, tokenizer=tokenizer, clear=False)

    predictor_collate_fun = functools.partial(collate_fun, tokenizer=tokenizer)
    predictor = Predictor(model, device,
                          collate_fun=predictor_collate_fun,
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
