import logging
from collections import defaultdict
import functools

import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, RobertaTokenizer

from model.model import BertForQuestionAnswering, Tokenizer, LabelSmoothingLossWithLogits, FocalLossWithLogits, WeightedLoss
from model.dataset import collate_fun, RawPreprocessor, SplitDataset, DummyDataset

logger = logging.getLogger(__name__)


def init_loss(params, train_weights):
    def _wght(name_):
        return getattr(params, name_, 1)

    if params.loss == 'ce':
        class_loss = nn.CrossEntropyLoss(weight=train_weights['label_weights'])
    elif params.loss == 'focal':
        class_loss = FocalLossWithLogits(alpha=params.focal_alpha, gamma=params.focal_gamma)
    elif params.loss == 'smooth':
        class_loss = LabelSmoothingLossWithLogits(n_classes=len(RawPreprocessor.labels2id),
                                                  smoothing=params.smooth_alpha)
    else:
        raise NotImplementedError

    logger.info(f'Used loss function for classification: {type(class_loss).__name__}.')

    init_losses = {'start_class': (nn.CrossEntropyLoss(ignore_index=-1), _wght('w_start')),
                   'end_class': (nn.CrossEntropyLoss(ignore_index=-1), _wght('w_end')),
                   'start_reg': (nn.MSELoss(), _wght('w_start_reg')),
                   'end_reg': (nn.MSELoss(), _wght('w_end_reg')),
                   'cls': (class_loss, _wght('w_cls'))}

    return WeightedLoss(init_losses)


def _load_checkpoint(model, checkpoint, *, device=torch.device('cpu')):
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict['model'], strict=False)

        logger.info(f'Model checkpoint was restored from {checkpoint}.')


def init_model(model_params, *, checkpoint=None, device=torch.device('cpu'), bpe_dropout=None):
    model_params.model_name = model_params.model.split('-')[0]

    # todo: https://github.com/huggingface/transformers/issues/2392
    model_params.model = './data/roberta' if model_params.model_name == 'roberta' else model_params.model

    if model_params.vocab_file is not None:
        tokenizer = Tokenizer(model_name=model_params.model_name,
                              vocab_file=model_params.vocab_file,
                              merges_file=model_params.merges_file,
                              lowercase=model_params.lowercase,
                              handle_chinese_chars=model_params.handle_chinese_chars,
                              dropout=bpe_dropout)
    else:
        if model_params.model_name == 'bert':
            tokenizer = BertTokenizer.from_pretrained(model_params.model)
        elif model_params.model_name == 'robrta':
            tokenizer = RobertaTokenizer.from_pretrained(model_params.model)
        else:
            raise NotImplementedError

        tokenizer.model_name = model_params.model_name

    model = BertForQuestionAnswering(model_params, num_labels=len(RawPreprocessor.labels2id))

    model.to(device)

    if checkpoint is not None:
        _load_checkpoint(model, checkpoint, device=device)

    return model, tokenizer


def init_datasets(params, *, tokenizer=None, clear=False):
    preprocessor = RawPreprocessor(raw_json=params.data_path,
                                   out_dir=params.processed_data_path,
                                   clear=clear)
    # dummy_dataset
    weights = defaultdict(lambda: None)

    if params.dummy_dataset:
        train_indexes = None
        test_indexes = None

        dataset_class = DummyDataset

        logger.warning('Dummy dataset is used to train model.')

    else:
        dataset_class = SplitDataset

        labels_counter, labels, (train_indexes, train_labels, test_indexes, test_labels) = preprocessor()

        if getattr(params, 'train_label_weights', False):
            label_weights = np.asarray([1 / labels_counter[k] for k in sorted(labels_counter.keys())])
            label_weights = label_weights / np.sum(label_weights)

            logger.info(f'Label weights: {", ".join([f"{RawPreprocessor.id2labels[k]} ({k}) - {v:.4f}" for k, v in enumerate(label_weights)])}.')

            weights['label_weights'] = torch.from_numpy(label_weights)

        if getattr(params, 'train_sampler_weights', False):
            sampler_weights = np.asarray([1 / (labels_counter[label]) for label in train_labels])
            sampler_weights = sampler_weights / np.sum(sampler_weights)

            weights['sampler_weights'] = sampler_weights

    train_dataset = dataset_class(data_dir=params.processed_data_path,
                                  tokenizer=tokenizer,
                                  indexes=train_indexes,
                                  max_seq_len=params.max_seq_len,
                                  max_question_len=params.max_question_len,
                                  doc_stride=params.doc_stride,
                                  split_by_sentence=params.split_by_sentence,
                                  truncate=params.truncate)
    test_dataset = dataset_class(data_dir=params.processed_data_path,
                                 tokenizer=tokenizer,
                                 indexes=test_indexes,
                                 test=True,
                                 max_seq_len=params.max_seq_len,
                                 max_question_len=params.max_question_len,
                                 doc_stride=params.doc_stride,
                                 split_by_sentence=params.split_by_sentence,
                                 truncate=params.truncate) \
        if getattr(params, 'local_rank', -1) in [-1, 0] else None

    return train_dataset, test_dataset, weights


def init_collate_fun(tokenizer, return_items=False):
    return functools.partial(collate_fun, tokenizer=tokenizer, return_items=return_items)
