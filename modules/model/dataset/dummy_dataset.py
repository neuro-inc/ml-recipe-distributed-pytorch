import numpy as np

from .split_dataset import DatasetItem


class DummyDataset:
    def __init__(self,
                 tokenizer,
                 *args,
                 max_seq_len=384,
                 max_question_len=64,
                 dataset_len=10000,
                 **kwargs):

        self.tokenizer = tokenizer
        self.dataset_len = dataset_len

        self.max_seq_len = max_seq_len
        self.max_question_len = max_question_len

        self.w_ids = [self.tokenizer.pad_token_id,
                      self.tokenizer.sep_token_id,
                      self.tokenizer.cls_token_id]

    def __len__(self):
        return self.dataset_len

    def _delete_special(self, ids):
        for w_id in self.w_ids:
            ids[ids == w_id] = self.tokenizer.unk_token_id

        return ids

    def __getitem__(self, *args):
        document_len = self.max_seq_len - self.max_question_len - 3

        question_ids = self._delete_special(np.random.randint(1, len(self.tokenizer), self.max_question_len)).tolist()
        document_ids = self._delete_special((np.random.randint(1, len(self.tokenizer), document_len))).tolist()

        input_ids = [self.tokenizer.cls_token_id] + question_ids + \
                    [self.tokenizer.sep_token_id] + document_ids + [self.tokenizer.sep_token_id]

        return DatasetItem(input_ids=input_ids,
                           start_id=0,
                           end_id=self.max_seq_len-1,
                           label_id=0,
                           example_id='None',
                           start_position=0,
                           end_position=1)
