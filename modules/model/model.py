import logging

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, RobertaModel, RobertaTokenizer

from .split_dataset import RawPreprocessor
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def _load_checkpoint(model, checkpoint, *, device=torch.device('cpu')):
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict['model'])

        logger.info(f'Model checkpoint was restored from {checkpoint}.')


def get_model(model_params, *, checkpoint=None, device=torch.device('cpu'), bpe_dropout=None):
    model = model_params.model
    model_name = model.split('-')[0]

    # todo: https://github.com/huggingface/transformers/issues/2392
    model = model if 'roberta' not in model else './data/roberta'

    tokenizer = Tokenizer(model_name=model_name,
                          vocab_file=model_params.vocab_file,
                          merges_file=model_params.merges_file,
                          lowercase=model_params.lowercase,
                          handle_chinese_chars=model_params.handle_chinese_chars,
                          dropout=bpe_dropout)

    model = BertForQuestionAnswering.from_pretrained(model,
                                                     hidden_dropout_prob=model_params.hidden_dropout_prob,
                                                     attention_probs_dropout_prob=model_params.attention_probs_dropout_prob,
                                                     layer_norm_eps=model_params.layer_norm_eps,
                                                     num_labels=len(RawPreprocessor.labels2id))
    model.to(device)

    if checkpoint is not None:
        _load_checkpoint(model, checkpoint, device=device)

    return model, tokenizer


class BertForQuestionAnswering(BertPreTrainedModel):
    """BERT model for QA and classification tasks.

    Parameters
    ----------
    config : transformers.BertConfig. Configuration class for BERT.

    Returns
    -------
    start_logits : torch.Tensor with shape (batch_size, sequence_size).
        Starting scores of each tokens.
    end_logits : torch.Tensor with shape (batch_size, sequence_size).
        Ending scores of each tokens.
    classifier_logits : torch.Tensor with shape (batch_size, num_classes).
        Classification scores of each labels.
    """
    def __init__(self, config):
        super().__init__(config)
        # todo: pass args to model
        self.transformer = RobertaModel(config) if config.type_vocab_size == 1 else BertModel(config)

        self.position_outputs = nn.Linear(config.hidden_size, 2)  # start/end

        self.classifier = nn.Sequential(nn.Dropout(config.hidden_dropout_prob),
                                        nn.Linear(config.hidden_size, config.num_labels))

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.transformer(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # predict start & end position
        position_logits = self.position_outputs(sequence_output)
        start_logits, end_logits = position_logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # classification
        classifier_logits = self.classifier(pooled_output)

        return start_logits, end_logits, classifier_logits
