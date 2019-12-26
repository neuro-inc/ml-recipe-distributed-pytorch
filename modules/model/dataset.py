import linecache
import os
import random
import re
import subprocess
from dataclasses import dataclass
from typing import List
import json


@dataclass
class DatasetItem(object):
    input_ids: List[int]
    start_id: int
    end_id: int
    label_id: int


class QADataloaderOptimal:
    def __init__(self, dataset_path, tokenizer, *,
                 max_seq_len=384,
                 max_question_len=64,
                 doc_stride=128
                 ):
        self.dataset_path = dataset_path
        assert os.path.exists(self.dataset_path), f'Dataset file {self.dataset_path} does not exist.'

        self.num_elements = int(subprocess.check_output('wc -l ' + self.dataset_path, shell=True).split()[0])
        assert self.num_elements > 0, f'Dataset file {self.dataset_path} does not contain anything.'

        self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len
        self.max_question_len = max_question_len
        self.doc_stride = doc_stride

        self.labels2id = {k: i for i, k in enumerate(['yes', 'no', 'short', 'long', 'unknown'])}
        self.id2labels = {i: k for k, i in self.labels2id.items()}

    def __len__(self):
        return self.num_elements

    def _load_line(self, indx):
        raw_line = linecache.getline(self.dataset_path, indx + 1)
        raw_line = json.loads(raw_line)

        line = {}

        document_text = raw_line['document_text'].split()

        line['document_text'] = raw_line['document_text']
        line['question_text'] = raw_line['question_text']

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

    def _drop_tags(self, text):
        text = text.split()

        o2t = []
        t2o = []

        tokenized_text = []
        for word_i, word in enumerate(text):
            o2t.append(len(tokenized_text))
            if re.match(r'<.+>', word):
                continue

            word_tokens = self.tokenizer.tokenize(word)
            for token in word_tokens:
                t2o.append(word_i)
                tokenized_text.append(token)

        return tokenized_text, o2t, t2o

    def __getitem__(self, indx):
        try:
            line = self._load_line(indx)

            tokenized_text, o2t, t2o = self._drop_tags(line['document_text'])
            tokenized_question = self.tokenizer.tokenize(line['question_text'])[:self.max_question_len]

            # targets: class_label, start, end
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
                                          label_id=self.labels2id[label]))

            item = random.sample(chunks, 1)[0]

        except Exception as e:
            raise e

        return item
