import logging

from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer

logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self, model_name, vocab_file, *,
                 merges_file=None,
                 lowercase=True,
                 handle_chinese_chars=False,
                 dropout=None):

        self.model_name = model_name

        if model_name == 'bert':
            self._pad_token = '[PAD]'
            self._sep_token = '[SEP]'
            self._cls_token = '[CLS]'
            self._unk_token = '[UNK]'

            if dropout is not None:
                logger.warning('BPE dropout is not supported by BertWordPieceTokenizer.')

            self.tokenizer = BertWordPieceTokenizer(vocab_file,
                                                    lowercase=lowercase,
                                                    handle_chinese_chars=handle_chinese_chars,
                                                    add_special_tokens=False,
                                                    unk_token=self.unk_token,
                                                    cls_token=self.cls_token,
                                                    sep_token=self.sep_token)
        elif model_name == 'roberta':
            if merges_file is None:
                raise AttributeError('To use ByteLevelTokenizer, specify path to merges file.')

            self._pad_token = '<pad>'
            self._sep_token = '</s>'
            self._cls_token = '<s>'
            self._unk_token = '<unk>'

            try:
                self.tokenizer = ByteLevelBPETokenizer(vocab_file=vocab_file,
                                                       merges_file=merges_file,
                                                       dropout=dropout)
            except TypeError as e:
                logger.warning('BPE dropout is not supported by ByteLevelBPETokenizer.')
                logger.error(e)
                self.tokenizer = ByteLevelBPETokenizer(vocab_file=vocab_file,
                                                       merges_file=merges_file)

        else:
            raise NotImplementedError(f'Tokenizer initialization for model {model_name} is not implemented.')

    def encode(self, string):
        return self.tokenizer.encode(string).ids

    @property
    def pad_token_id(self):
        return self.tokenizer.token_to_id(self._pad_token)

    @property
    def sep_token_id(self):
        return self.tokenizer.token_to_id(self._sep_token)

    @property
    def cls_token_id(self):
        return self.tokenizer.token_to_id(self._cls_token)

    @property
    def unk_token_id(self):
        return self.tokenizer.token_to_id(self._unk_token)

    @property
    def pad_token(self):
        return self._pad_token

    @property
    def sep_token(self):
        return self._sep_token

    @property
    def cls_token(self):
        return self._cls_token

    @property
    def unk_token(self):
        return self._unk_token
