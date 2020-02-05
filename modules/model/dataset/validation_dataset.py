import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import nltk

from .split_dataset import RawPreprocessor

logger = logging.getLogger(__name__)


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

    start_position: float
    end_position: float


class ChunkDataset:
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
        word_i = 0
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
        encoded_text, o2t, t2o, _, _ = ChunkDataset._drop_tags_and_encode(self.tokenizer, line['document_text'])
        encoded_question = self.tokenizer.encode(line['question_text'])[:self.max_question_len]

        class_label, start_position, end_position = RawPreprocessor._get_target(line)

        assert start_position <= end_position, 'Before mapping.'

        start_position = o2t[start_position]
        end_position = o2t[end_position]

        assert start_position <= end_position, 'After mapping.'

        example_id = line['example_id']

        document_len = self.max_seq_len - len(encoded_question) - 3  # [CLS], [SEP], [SEP]

        samples = []

        for doc_start in range(0, len(encoded_text), self.doc_stride):
            doc_end = doc_start + document_len
            if not (doc_start <= start_position and end_position <= doc_end):
                start, end, label = -1, -1, 'unknown'
            else:
                start = start_position - doc_start + len(encoded_question) + 2
                end = end_position - doc_start + len(encoded_question) + 2
                label = class_label

            samples.append((start, end, label, doc_start, doc_end))

            if self.test:
                break

        chunks = []
        for sample in samples:

            start, end, label, doc_start, doc_end = sample

            chunk = encoded_text[doc_start: doc_end]
            input_ids = [self.tokenizer.cls_token_id] + encoded_question + \
                        [self.tokenizer.sep_token_id] + chunk + \
                        [self.tokenizer.sep_token_id]

            assert -1 <= start <= self.max_seq_len, f'Incorrect start index: {start}.'
            assert -1 <= end <= self.max_seq_len, f'Incorrect start index: {end}.'

            chunks.append(ChunkItem(input_ids=input_ids,
                                    start_id=start,
                                    end_id=end,
                                    label_id=self.labels2id[label],  # self.labels2id[class_label],
                                    item_id=example_id,
                                    true_text=line['document_text'],
                                    true_question=line['question_text'],
                                    question_len=len(encoded_question),
                                    t2o=t2o,
                                    chunk_start=doc_start,
                                    chunk_end=doc_end,
                                    true_label=self.labels2id[class_label],
                                    true_start=start_position,
                                    true_end=end_position,
                                    start_position=start / self.max_seq_len,
                                    end_position=end / self.max_seq_len))

        return chunks

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
            sen_, o2t_, t2o_, hystory, start = ChunkDataset._drop_tags_and_encode(self.tokenizer, sen,
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

        samples.append((sample_ids, start, end, label, len(chunk_sents), doc_start, doc_end))

        chunks = []
        for sample in samples:
            sample_ids, start, end, label, cl, ds, de = sample

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

            chunks.append(ChunkItem(input_ids=input_ids,
                                    start_id=start,
                                    end_id=end,
                                    label_id=self.labels2id[label],
                                    item_id=example_id,
                                    true_text=line['document_text'],
                                    true_question=line['question_text'],
                                    question_len=len(encoded_question),
                                    t2o=t2o,
                                    chunk_start=doc_start,
                                    chunk_end=doc_end,
                                    true_label=self.labels2id[class_label],
                                    true_start=start_position,
                                    true_end=end_position,
                                    start_position=start / self.max_seq_len,
                                    end_position=end / self.max_seq_len))

        return chunks

    def __getitem__(self, idx):
        idx = self.indexes[idx]

        with open(self.data_dir / f'{idx}.json', 'r') as in_file:
            line = json.load(in_file)
        if self.split_by_sentence:
            chunks = self._split_doc_by_sentence(line)
        else:
            chunks = self._split_doc(line)

        return chunks
