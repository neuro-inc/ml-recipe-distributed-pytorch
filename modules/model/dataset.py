import dataclasses
import json
import linecache
import logging
import multiprocessing as mp
import os
import random
import re
import subprocess
from dataclasses import dataclass
from typing import List

from tqdm.auto import tqdm

logger = logging.getLogger(__file__)


@dataclass
class DatasetItem(object):
    example_id: str
    input_ids: List[int]
    start_id: int
    end_id: int
    label_id: int


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


class RawDataPreprocessor(object):
    labels2id = {k: i for i, k in enumerate(['yes', 'no', 'short', 'long', 'unknown'])}
    id2labels = {i: k for k, i in labels2id.items()}

    def __init__(self, in_path, out_path, tokenizer, *,
                 max_seq_len=384,
                 max_question_len=64,
                 doc_stride=128,
                 limit_data=0,
                 num_threads=2,
                 ):
        self.in_path = in_path
        self.out_path = out_path

        self.tokenizer = tokenizer

        self.data_extractor = LineDataExtractor(self.in_path)

        self.max_seq_len = max_seq_len
        self.max_question_len = max_question_len
        self.doc_stride = doc_stride

        self.limit_data = limit_data if limit_data else len(self.data_extractor)
        logger.info(f'{self.limit_data} lines will be processed.')

        self.num_threads = num_threads

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

    def _split_doc(self, line):
        tokenized_text, o2t, t2o = RawDataPreprocessor._drop_tags(self.tokenizer, line['document_text'])
        tokenized_question = self.tokenizer.tokenize(line['question_text'])[:self.max_question_len]

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

        start_position = o2t[start_position]
        end_position = o2t[end_position]

        example_id = line['example_id']

        document_len = self.max_seq_len - len(tokenized_question) - 3  # [CLS], [SEP], [SEP]

        chunks = []

        for doc_start in range(0, len(tokenized_text), self.doc_stride):
            doc_end = doc_start + document_len

            if doc_start <= start_position and end_position <= doc_end:
                start = start_position - doc_start + len(tokenized_question) + 2
                end = end_position - doc_start + len(tokenized_question) + 2
                label = class_label
            else:
                start = -1
                end = -1
                label = 'unknown'

            assert -1 <= start <= self.max_seq_len, f'Incorrect start index: {start}.'
            assert -1 <= end <= self.max_seq_len, f'Incorrect start index: {end}.'

            chunk_text = tokenized_text[doc_start: doc_end]
            input_tokens = [self.tokenizer.cls_token] + tokenized_question + \
                           [self.tokenizer.sep_token] + chunk_text + \
                           [self.tokenizer.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            chunks.append(DatasetItem(input_ids=input_ids,
                                      start_id=start,
                                      end_id=end,
                                      label_id=self.labels2id[label],
                                      example_id=example_id))

        return chunks

    def listener_fun(self, pool_queue):
        with open(self.out_path, 'w') as out_file:
            while True:
                chunk = pool_queue.get()

                if chunk == 'done':
                    break

                out_file.write(chunk)
                out_file.flush()

    def process_fun(self, pool_queue, idx):
        line = self.data_extractor[idx]
        line = RawDataPreprocessor._process_line(line)
        chunks = self._split_doc(line)

        for chunk in chunks:
            pool_queue.put(json.dumps(dataclasses.asdict(chunk)) + '\n')

    def __call__(self):
        manager = mp.Manager()
        pool_queue = manager.Queue()
        pool = mp.Pool(self.num_threads)

        pool.apply_async(self.listener_fun, (pool_queue,))

        tqdm_data = tqdm(total=self.limit_data)

        def display_update(*args, **kwargs):
            tqdm_data.update()

        jobs = []
        for i in range(self.limit_data):
            jobs.append(pool.apply_async(self.process_fun, (pool_queue, i), callback=display_update))

        for job in jobs:
            job.get()

        pool_queue.put('done')
        pool.close()
        pool.join()


class QADataloader:
    def __init__(self, dataset_path, *, indexes=None):
        self.dataset_path = dataset_path
        self.data_extractor = LineDataExtractor(self.dataset_path)

        self.labels2id = RawDataPreprocessor.labels2id
        self.id2labels = RawDataPreprocessor.id2labels

        self.indexes = indexes

    def __len__(self):
        return len(self.data_extractor) if self.indexes is None else len(self.indexes)

    def __getitem__(self, indx):
        indx = indx if self.indexes is None else self.indexes[indx]
        try:
            item = DatasetItem(**self.data_extractor[indx])
        except Exception as e:
            raise e

        return item


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.getLogger('transformers').setLevel('CRITICAL')
    logger = logging.getLogger(__file__)

    dataset = QADataloader('./data/preprocessed_train_small.jsonl')
    print(dataset[0])
