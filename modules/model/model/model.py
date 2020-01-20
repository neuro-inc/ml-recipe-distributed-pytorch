import logging

import torch.nn as nn
from transformers import BertModel, RobertaModel

logger = logging.getLogger(__name__)


MODELS = {'bert': BertModel,
          'roberta': RobertaModel}


class BertForQuestionAnswering(nn.Module):
    """BERT model for QA and classification tasks."""
    def __init__(self, model_params, num_labels):
        super().__init__()

        self.model_params = model_params

        self.transformer = MODELS[model_params.model_name].\
            from_pretrained(model_params.model,
                            hidden_dropout_prob=model_params.hidden_dropout_prob,
                            attention_probs_dropout_prob=model_params.attention_probs_dropout_prob,
                            layer_norm_eps=model_params.layer_norm_eps,
                            num_labels=num_labels)

        config = self.transformer.config

        # if self.model_params.out_class_pos:
        self.position_outputs = nn.Linear(config.hidden_size, 2)  # start/end

        # if self.model_params.out_chunk_class:
        self.classifier = nn.Sequential(nn.Dropout(config.hidden_dropout_prob),
                                        nn.Linear(config.hidden_size, config.num_labels))

        # if self.model_params.out_reg_pos:
        self.reg_start = nn.Sequential(nn.Linear(config.hidden_size, 1),
                                       nn.Sigmoid())

        self.reg_end = nn.Sequential(nn.Linear(config.hidden_size, 1),
                                     nn.Sigmoid())

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

        # regression
        reg_start = self.reg_start(pooled_output).squeeze(-1)
        reg_end = self.reg_end(pooled_output).squeeze(-1)

        result = {'start_class': start_logits,
                  'end_class': end_logits,
                  'start_reg': reg_start,
                  'end_reg': reg_end,
                  'cls': classifier_logits}

        return result
