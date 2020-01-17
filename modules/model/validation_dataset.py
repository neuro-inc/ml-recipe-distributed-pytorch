import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .split_dataset import RawPreprocessor


@dataclass
class ChunkItem(object):
    item_id: str
    input_ids: List[int]
    start_id: int
    end_id: int
    label_id: int

    true_text: str
    true_question: str
    true_label: int
    true_start: int
    true_end: int

    question_len: int

    t2o: List[int]

    chunk_start: int
    chunk_end: int

    question_len: int


class ChunkDataset:
    def __init__(self, data_dir, tokenizer, indexes, *,
                 max_seq_len=384,
                 max_question_len=64,
                 doc_stride=32):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.indexes = indexes

        self.max_seq_len = max_seq_len
        self.max_question_len = max_question_len
        self.doc_stride = doc_stride

        self.labels2id = RawPreprocessor.labels2id
        self.id2labels = RawPreprocessor.id2labels

        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)

    def __len__(self):
        return len(self.indexes)

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
        tokenized_text, o2t, t2o = ChunkDataset._drop_tags(self.tokenizer, line['document_text'])
        tokenized_question = self.tokenizer.tokenize(line['question_text'])[:self.max_question_len]

        class_label, start_position, end_position = RawPreprocessor._get_target(line)

        start_position = o2t[start_position]
        end_position = o2t[end_position]

        example_id = line['example_id']

        document_len = self.max_seq_len - len(tokenized_question) - 3  # [CLS], [SEP], [SEP]

        chunks = []
        for doc_start in range(0, len(tokenized_text), self.doc_stride):
            doc_end = doc_start + document_len
            if not (doc_start <= start_position and end_position <= doc_end):
                start, end, label = -1, -1, 'unknown'
            else:
                start = start_position - doc_start + len(tokenized_question) + 2
                end = end_position - doc_start + len(tokenized_question) + 2
                label = class_label

            chunk_text = tokenized_text[doc_start: doc_end]
            input_tokens = [self.tokenizer.cls_token] + tokenized_question + \
                           [self.tokenizer.sep_token] + chunk_text + \
                           [self.tokenizer.sep_token]

            end = min(end, len(input_tokens) - 1)

            assert -1 <= start <= self.max_seq_len, f'Incorrect start index: {start}.'
            assert -1 <= end <= self.max_seq_len, f'Incorrect start index: {end}.'

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            chunks.append(ChunkItem(input_ids=input_ids,
                                    start_id=start,
                                    end_id=end,
                                    label_id=self.labels2id[class_label],#self.labels2id[label],
                                    item_id=example_id,
                                    true_text=line['document_text'],
                                    true_question=line['question_text'],
                                    question_len=len(tokenized_question),
                                    t2o=t2o,
                                    chunk_start=doc_start,
                                    chunk_end=doc_end,
                                    true_label=self.labels2id[class_label],
                                    true_start=start_position,
                                    true_end=end_position
            ))

        return chunks

    def __getitem__(self, idx):
        idx = self.indexes[idx]

        with open(self.data_dir / f'{idx}.json', 'r') as in_file:
            line = json.load(in_file)

        chunk = self._split_doc(line)

        return chunk
