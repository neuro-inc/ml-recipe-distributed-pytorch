import json
import linecache
import logging
import os
import pickle
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List

import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

logger = logging.getLogger(__file__)


class LineDataExtractor(object):
    def __init__(self, data_path):
        self.data_path = data_path

        logger.info(f'Loading number of lines in file {self.data_path}...')
        self.num_lines = int(subprocess.check_output('wc -l ' + self.data_path, shell=True).split()[0])
        logger.info(f'Line number is {self.num_lines}.')

        self.line_i = 0

    def __len__(self):
        return self.num_lines

    def __iter__(self):
        return self

    def __next__(self):
        if self.line_i < self.num_lines:
            line = json.loads(linecache.getline(self.data_path, self.line_i + 1))
            self.line_i += 1
            return line
        else:
            raise StopIteration

    def __getitem__(self, idx):
        return json.loads(linecache.getline(self.data_path, idx + 1))


class RawPreprocessor(object):
    labels2id = {k: i for i, k in enumerate(['yes', 'no', 'short', 'long', 'unknown'])}
    id2labels = {i: k for k, i in labels2id.items()}

    def __init__(self, raw_json, out_dir, *, clear=False):

        self.raw_json = raw_json
        self.out_dir = out_dir

        if isinstance(self.out_dir, str):
            self.out_dir = Path(self.out_dir)

        os.makedirs(self.out_dir, exist_ok=True)

        self.data_extractor = LineDataExtractor(self.raw_json)

        self.label_info_path = self.out_dir / 'label.info'
        self.split_info_path = self.out_dir / 'split.info'

        if clear:
            rm_files = self.out_dir.glob('*')
            for rm_file in rm_files:
                os.remove(rm_file)

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

        # long_answer_candidates = []
        # for d in raw_line['long_answer_candidates']:
        #     if d['top_level']:
        #         start = d['start_token']
        #         end = d['end_token']
        #
        #         long_answer_candidates.append(' '.join(document_text[start:end]))
        # line['long_answer_candidates'] = long_answer_candidates

        line['long_answer_candidates'] = raw_line['long_answer_candidates']

        return line

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

    def __call__(self):
        if self.label_info_path.exists():
            with open(self.label_info_path, 'rb') as in_file:
                labels_counter, labels = pickle.load(in_file)
            logging.info(f'Labels info was loaded from {self.label_info_path}.')
        else:
            labels_counter = defaultdict(int)
            labels = np.zeros((len(self.data_extractor, )))

            tqdm_data = tqdm(self.data_extractor, desc='Processing lines')

            for line_i, line in enumerate(tqdm_data):
                line = RawPreprocessor._process_line(line)

                label = self.labels2id[RawPreprocessor._get_target(line)[0]]

                labels[line_i] = label
                labels_counter[label] += 1

                out_path = self.out_dir / f'{line_i}.json'

                with open(out_path, 'w') as out_file:
                    json.dump(line, out_file)

            with open(self.label_info_path, 'wb') as out_file:
                pickle.dump((labels_counter, labels), out_file)
                logger.info(f'Label information was dumped to {self.label_info_path}')

        split_info = self._split_train_test(labels)

        return labels_counter, labels, split_info

    def _split_train_test(self, labels):
        if self.split_info_path.exists():
            with open(self.split_info_path, 'rb') as in_file:
                (train_indexes, train_labels, test_indexes, test_labels) = pickle.load(in_file)
            logger.info(f'Split information was loaded form {self.split_info_path}.')
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

            with open(self.split_info_path, 'wb') as out_file:
                pickle.dump((train_indexes, train_labels, test_indexes, test_labels), out_file)
                logger.info(f'Split information was dumped to {self.split_info_path}.')

        assert len(train_indexes) == len(train_labels)
        assert len(test_indexes) == len(test_labels)

        return train_indexes, train_labels, test_indexes, test_labels


@dataclass
class DatasetItem(object):
    example_id: str
    input_ids: List[int]
    start_id: int
    end_id: int
    label_id: int


class SplitDataset:
    def __init__(self, data_dir, tokenizer, indexes, *,
                 max_seq_len=384,
                 max_question_len=64,
                 doc_stride=128,
                 test=False,
                 split_by_sentence=False,
                 truncate=False):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.indexes = indexes

        self.max_seq_len = max_seq_len
        self.max_question_len = max_question_len
        self.doc_stride = doc_stride

        self.labels2id = RawPreprocessor.labels2id
        self.id2labels = RawPreprocessor.id2labels

        self.label2weight = {l: w for l, w in zip(['yes', 'no', 'short', 'long', 'unknown'], [1, 1, 1, 1, 1e-3])}

        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)

        self.test = test
        self.truncate = truncate

        self.split_by_sentence = split_by_sentence
        if self.split_by_sentence:
            logger.info(f'Documents will be split by sentence.')
            try:
                self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            except Exception as e:
                logger.warning('Data for sentence tokenizer will be downloaded. To prevent it in future, '
                               'download the required package in your env with: '
                               '\n\t import nltk \n\t nltk.download("punkt")')

                nltk.download('punkt')

                self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def __len__(self):
        return len(self.indexes)

    @staticmethod
    def _drop_tags_and_encode(tokenizer, text, *, history_len=0, start=-1):
        text = text.split()

        o2t = []
        t2o = []

        tokenized_text = []
        for word_i, word in enumerate(text, start=start + 1):
            o2t.append(len(tokenized_text) + history_len)
            if re.match(r'<.+>', word):
                continue

            word_tokens = tokenizer.encode(word)
            for token in word_tokens:
                t2o.append(word_i)
                tokenized_text.append(token)

        return tokenized_text, o2t, t2o, history_len + len(tokenized_text), word_i

    def _split_doc(self, line):
        encoded_text, o2t, t2o, _, _ = SplitDataset._drop_tags_and_encode(self.tokenizer, line['document_text'])
        encoded_question = self.tokenizer.encode(line['question_text'])[:self.max_question_len]

        class_label, start_position, end_position = RawPreprocessor._get_target(line)

        assert start_position <= end_position, 'Before mapping.'

        start_position = o2t[start_position]
        end_position = o2t[end_position]

        assert start_position <= end_position, 'After mapping.'

        example_id = line['example_id']

        document_len = self.max_seq_len - len(encoded_question) - 3  # [CLS], [SEP], [SEP]

        samples = []
        weights = []

        for doc_start in range(0, len(encoded_text), self.doc_stride):
            doc_end = doc_start + document_len
            if not (doc_start <= start_position and end_position <= doc_end):
                start, end, label = -1, -1, 'unknown'
            else:
                start = start_position - doc_start + len(encoded_question) + 2
                end = end_position - doc_start + len(encoded_question) + 2
                label = class_label

            weights.append(self.label2weight[label])
            samples.append((start, end, label, doc_start, doc_end))

            if self.test:
                break

        weights = np.asarray(weights)
        weights = weights / np.sum(weights)

        idx = np.random.choice(np.arange(len(samples)), 1, p=weights)[0]
        start, end, label, doc_start, doc_end = samples[idx]

        chunk = encoded_text[doc_start: doc_end]
        input_ids = [self.tokenizer.cls_token_id] + encoded_question + \
                    [self.tokenizer.sep_token_id] + chunk + \
                    [self.tokenizer.sep_token_id]

        assert -1 <= start <= self.max_seq_len, f'Incorrect start index: {start}.'
        assert -1 <= end <= self.max_seq_len, f'Incorrect start index: {end}.'

        return DatasetItem(input_ids=input_ids,
                           start_id=start,
                           end_id=end,
                           label_id=self.labels2id[label],
                           example_id=example_id)

    def _split_doc_by_sentence(self, line):
        text = line['document_text']
        question = line['question_text']

        encoded_question = self.tokenizer.encode(question)[:self.max_question_len]
        document_len = self.max_seq_len - len(encoded_question) - 3

        class_label, start_position, end_position = RawPreprocessor._get_target(line)

        sens = self.sentence_tokenizer.tokenize(text)

        example_id = line['example_id']

        # sentences encoding
        t_sens, t_o2t, t_t2o = [], [], []
        start = -1
        hystory = 0
        for sen in sens:
            sen_, o2t_, t2o_, hystory, start = SplitDataset._drop_tags_and_encode(self.tokenizer, sen,
                                                                                  history_len=hystory,
                                                                                  start=start)

            t_sens.append(sen_)
            t_o2t.append(o2t_)
            t_t2o.append(t2o_)

        # maping positions into token positions
        o2t = sum(t_o2t, [])
        t2o = sum(t_t2o, [])

        assert start_position <= end_position, 'Before mapping.'

        o_sp, o_ep = start_position, end_position

        start_position = o2t[start_position]
        end_position = o2t[end_position]

        assert start_position <= end_position, 'After mapping.'

        # chunk initialization
        doc_start = 0
        doc_end = 0

        chunk_sents = []
        chunk_o2t = []
        chunk_t2o = []

        samples = []
        weights = []

        for sen_, o2t_, t2o_ in zip(t_sens, t_o2t, t_t2o):
            assert doc_end - doc_start >= 0

            if doc_end - doc_start + len(sen_) > document_len:
                while len(chunk_sents) and (doc_end - doc_start + len(sen_) > document_len):
                    sample_ids = sum(chunk_sents, [])
                    if not (doc_start <= start_position and end_position <= doc_end):
                        start, end, label = -1, -1, 'unknown'
                    else:
                        start = start_position - doc_start + len(encoded_question) + 2
                        end = end_position - doc_start + len(encoded_question) + 2
                        label = class_label

                    weights.append(self.label2weight[label])
                    samples.append((sample_ids, start, end, label, len(chunk_sents), doc_start, doc_end))

                    del_sen = chunk_sents.pop(0)

                    chunk_o2t.pop(0)
                    chunk_t2o.pop(0)
                    doc_start += len(del_sen)

            doc_end += len(sen_)

            chunk_sents.append(sen_)
            chunk_o2t.append(o2t_)
            chunk_t2o.append(t2o_)

        # add tail
        sample_ids = sum(chunk_sents, [])
        if not (doc_start <= start_position and end_position <= doc_end):
            start, end, label = -1, -1, 'unknown'
        else:
            start = start_position - doc_start + len(encoded_question) + 2
            end = end_position - doc_start + len(encoded_question) + 2
            label = class_label

        weights.append(self.label2weight[label])
        samples.append((sample_ids, start, end, label, len(chunk_sents), doc_start, doc_end))

        # sampling
        assert len(samples), f'Empty document: {example_id}?'

        if self.test:
            for idx in range(len(samples)):
                label = samples[idx][3]
                if label == class_label:
                    break
        else:
            weights = np.asarray(weights)
            weights = weights / np.sum(weights)

            idx = np.random.choice(np.arange(len(samples)), 1, p=weights)[0]

        sample_ids, start, end, label, cl, ds, de = samples[idx]

        if self.truncate and len(sample_ids) > document_len:
            start_ = start - len(encoded_question) - 2
            end_ = end - len(encoded_question) - 2

            if start_ < document_len and end_ < document_len:
                sample_ids = sample_ids[:document_len]
            else:
                sample_ids = sample_ids[start_:start_ + document_len]
                start_ = 0
                end_ = min(end_ - start_, len(sample_ids))

                start = start_ + len(encoded_question) + 2
                end = end_ + len(encoded_question) + 2

        input_ids = [self.tokenizer.cls_token_id] + encoded_question + \
                    [self.tokenizer.sep_token_id] + sample_ids + \
                    [self.tokenizer.sep_token_id]

        assert len(input_ids) <= self.max_seq_len, f'Chunk length {len(input_ids)} : ' \
                                                   f'(#chunks {cl} - s: {start}: e: {end} | ' \
                                                   f'ds: {ds}: de: {de} | ts: {len(t2o)} ' \
                                                   f'is more then limit {self.max_seq_len}. Label: {label}. \n' \
                                                   f'Question: {question} \n' \
                                                   f'Text: {self.tokenizer.decode(sample_ids)} \n' \
                                                   f'Answer: {self.tokenizer.decode(input_ids[start:end])} \n' \
                                                   f'Original answer: {" ".join(text.split()[o_sp: o_ep])}'
        assert -1 <= start < self.max_seq_len, f'Incorrect start index: {start}.'
        assert -1 <= end < self.max_seq_len, f'Incorrect start index: {end}.'

        return DatasetItem(input_ids=input_ids,
                           start_id=start,
                           end_id=end,
                           label_id=self.labels2id[label],
                           example_id=example_id)

    def __getitem__(self, idx):
        idx = self.indexes[idx]

        with open(self.data_dir / f'{idx}.json', 'r') as in_file:
            line = json.load(in_file)
        if self.split_by_sentence:
            chunk = self._split_doc_by_sentence(line)
        else:
            chunk = self._split_doc(line)

        return chunk
