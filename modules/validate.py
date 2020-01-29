import multiprocessing as mp

import torch

from utils import get_logger, set_seed, show_params
from init import init_collate_fun, init_model

from model.utils.parser import get_model_parser, get_predictor_parser, get_params
from model.inference.predictor import Predictor
from model.dataset import RawPreprocessor, ChunkDataset

from transformers import BertTokenizer


def get_validation_dataset(params, *, tokenizer=None, clear=False):
    preprocessor = RawPreprocessor(raw_json=params.data_path,
                                   out_dir=params.processed_data_path,
                                   clear=clear)
    _, _, (_, _, val_indexes, val_labels) = preprocessor()

    val_dataset = ChunkDataset(params.processed_data_path, tokenizer, val_indexes,
                               test=False,
                               split_by_sentence=True,
                               truncate=True)

    return val_dataset


def main(params, model_params):
    show_params(model_params, 'model')
    show_params(params, 'predictor')

    device = torch.device('cuda') if torch.cuda.is_available() and params.gpu else torch.device('cpu')

    model, tokenizer = init_model(model_params, checkpoint=params.checkpoint, device=device)

    # todo: Tokenizer from tokenizers does not work with my implementation of dataloader
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
    # predictor.show_predictions()

    return predictor


if __name__ == '__main__':
    _, (params, model_params) = get_params((get_predictor_parser, get_model_parser))
    logger = get_logger(logger_name='validate')

    params.n_jobs = min(params.n_jobs, mp.cpu_count() // 2)

    main(params, model_params)
