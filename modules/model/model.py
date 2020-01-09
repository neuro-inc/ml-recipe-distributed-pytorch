import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


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
        super(BertForQuestionAnswering, self).__init__(config)
        self.transformer = BertModel(config)

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
