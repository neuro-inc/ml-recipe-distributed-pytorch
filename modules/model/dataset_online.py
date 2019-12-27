import logging
import os
import pickle
import random
import re
from collections import defaultdict

import numpy as np
from .dataset import LineDataExtractor, DatasetItem
from sklearn.model_selection import train_test_split
from tqdm.auto import trange

logger = logging.getLogger(__file__)


class DataPreprocessorOnline(object):
    labels2id = {k: i for i, k in enumerate(['yes', 'no', 'short', 'long', 'unknown'])}
    id2labels = {i: k for k, i in labels2id.items()}

    def __init__(self, in_path, *,
                 label_info_dump=None,
                 split_info_dump=None):
        self.in_path = in_path

        self.data_extractor = LineDataExtractor(self.in_path)

        self.label_info_dump = label_info_dump
        self.split_info_dump = split_info_dump

    def scan_labels(self):
        if self.label_info_dump is not None and os.path.exists(self.label_info_dump):
            with open(self.label_info_dump, 'rb') as in_file:
                label_counter, labels = pickle.load(in_file)
            logging.info(f'Label info was loaded from {self.label_info_dump}.')
        else:
            label_counter = defaultdict(int)
            labels = np.zeros((len(self.data_extractor, )))

            for label_i in trange(len(self.data_extractor), desc='Loading labels'):

                line = DataPreprocessorDatasetOnline._process_line(self.data_extractor[label_i])

                label = self.labels2id[DataPreprocessorDatasetOnline._get_target(line)[0]]

                labels[label_i] = label
                label_counter[label] += 1

            if self.label_info_dump is not None:
                with open(self.label_info_dump, 'wb') as out_file:
                    pickle.dump((label_counter, labels), out_file)
                logger.info(f'Label information was dumped to {self.label_info_dump}')

        return label_counter, labels

    def split_train_test(self, labels):
        if self.split_info_dump is not None and os.path.exists(self.split_info_dump):
            with open(self.split_info_dump, 'rb') as in_file:
                (train_indexes, train_labels, test_indexes, test_labels) = pickle.load(in_file)
            logger.info(f'Split information was loaded form {self.split_info_dump}.')
        else:
            indexes = np.arange(len(labels))

            train_indexes, train_labels, test_indexes, test_labels = [], [], [], []
            for label_i in range(len(self.labels2id)):
                mask = (labels == label_i)

                train_ids, test_ids, train_l, test_l = train_test_split(indexes[mask], labels[mask],
                                                                        test_size=0.05, random_state=0)

                train_indexes.append(train_ids)
                train_labels.append(train_l)
                test_indexes.append(test_ids)
                test_labels.append(test_l)

            train_indexes = np.concatenate(train_indexes, axis=0)
            train_labels = np.concatenate(train_labels, axis=0)
            test_indexes = np.concatenate(test_indexes, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)

            if self.split_info_dump is not None:
                with open(self.split_info_dump, 'wb') as out_file:
                    pickle.dump((train_indexes, train_labels, test_indexes, test_labels), out_file)
                logger.info(f'Split information was dumped to {self.split_info_dump}.')

        assert len(train_indexes) == len(train_labels)
        assert len(test_indexes) == len(test_labels)

        return train_indexes, train_labels, test_indexes, test_labels


class DataPreprocessorDatasetOnline:

    def __init__(self, dataprocessor, tokenizer, indexes, *,
                 max_seq_len=384,
                 max_question_len=64,
                 doc_stride=128):

        self.dataprocessor = dataprocessor
        self.tokenizer = tokenizer
        self.indexes = indexes

        self.max_seq_len = max_seq_len
        self.max_question_len = max_question_len
        self.doc_stride = doc_stride

        self.labels2id = DataPreprocessorOnline.labels2id
        self.id2labels = DataPreprocessorOnline.id2labels

    def __len__(self):
        return len(self.indexes)

    @staticmethod
    def _process_line(raw_line):
        line = {}

        document_text = raw_line['document_text'].split()

        line['document_text'] = raw_line['document_text']
        line['question_text'] = raw_line['question_text']
        line['example_id'] = raw_line['example_id']

        annotations = raw_line['annotations'][0]

        line['yes_no_answer'] = annotations['yes_no_answer']

        start = annotations['long_answer']['start_token']
        end = annotations['long_answer']['end_token']
        line['long_answer'] = 'NONE' if start == end else document_text[start:end]
        line['long_answer_start'] = start
        line['long_answer_end'] = end
        line['long_answer_index'] = annotations['long_answer']['candidate_index']

        line['short_answers'] = annotations['short_answers']

        long_answer_candidates = []
        for d in raw_line['long_answer_candidates']:
            if d['top_level']:
                start = d['start_token']
                end = d['end_token']

                long_answer_candidates.append(' '.join(document_text[start:end]))

        line['long_answer_candidates'] = long_answer_candidates

        return line

    @staticmethod
    def _drop_tags(tokenizer, text):
        text = text.split()

        o2t = []
        t2o = []

        tokenized_text = []
        for word_i, word in enumerate(text):
            o2t.append(len(tokenized_text))
            if re.match(r'<.+>', word):
                continue

            word_tokens = tokenizer.tokenize(word)
            for token in word_tokens:
                t2o.append(word_i)
                tokenized_text.append(token)

        return tokenized_text, o2t, t2o

    @staticmethod
    def _get_target(line):
        if line['yes_no_answer'] in ['YES', 'NO']:
            class_label = line['yes_no_answer'].lower()
            start_position = line['long_answer_start']
            end_position = line['long_answer_end']
        elif line['short_answers']:
            class_label = 'short'
            # todo: find optimal interval
            short_answers = line['short_answers']
            start_position = short_answers[0]['start_token']
            end_position = short_answers[0]['end_token']
        elif line['long_answer_index'] != -1:
            class_label = 'long'
            start_position = line['long_answer_start']
            end_position = line['long_answer_end']
        else:
            class_label = 'unknown'
            start_position = -1
            end_position = -1

        return class_label, start_position, end_position

    def _split_doc(self, line):
        tokenized_text, o2t, t2o = DataPreprocessorDatasetOnline._drop_tags(self.tokenizer, line['document_text'])
        tokenized_question = self.tokenizer.tokenize(line['question_text'])[:self.max_question_len]

        class_label, start_position, end_position = DataPreprocessorDatasetOnline._get_target(line)

        start_position = o2t[start_position]
        end_position = o2t[end_position]

        example_id = line['example_id']

        document_len = self.max_seq_len - len(tokenized_question) - 3  # [CLS], [SEP], [SEP]

        if class_label == DataPreprocessorOnline.labels2id['short'] or class_label == DataPreprocessorOnline.labels2id['long']:
            ans_distance = min(end_position - start_position, document_len)

            doc_start = random.randint(max(start_position - ans_distance, 0), start_position)
            doc_end = doc_start + document_len

            start = start_position - doc_start + len(tokenized_question) + 2
            end = end_position - doc_start + len(tokenized_question) + 2
            label = class_label

        else:
            start = -1
            end = -1
            label = class_label

            doc_start = random.randint(0, max(len(tokenized_text) - document_len, 1))
            doc_end = doc_start + document_len

        chunk_text = tokenized_text[doc_start: doc_end]
        input_tokens = [self.tokenizer.cls_token] + tokenized_question + \
                       [self.tokenizer.sep_token] + chunk_text + \
                       [self.tokenizer.sep_token]

        end = min(end, len(input_tokens) - 1)

        assert -1 <= start <= self.max_seq_len, f'Incorrect start index: {start}.'
        assert -1 <= end <= self.max_seq_len, f'Incorrect start index: {end}.'

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        return DatasetItem(input_ids=input_ids,
                           start_id=start,
                           end_id=end,
                           label_id=DataPreprocessorOnline.labels2id[label],
                           example_id=example_id)

    def __getitem__(self, idx):
        idx = self.indexes[idx]

        line = self.dataprocessor.data_extractor[idx]
        line = DataPreprocessorDatasetOnline._process_line(line)
        chunk = self._split_doc(line)

        return chunk