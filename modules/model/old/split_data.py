import logging

from model.dataset_online import DataPreprocessorOnline, DataPreprocessorDatasetOnline
from transformers import BertTokenizer


def main():
    bert_model = 'bert-base-uncased'
    do_lower_case = 'uncased' in bert_model

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    preprocessor = DataPreprocessorOnline('./data/simplified-nq-train.jsonl',
                                          label_info_dump='./data/train_labels.pkl',
                                          split_info_dump='./data/train_split.pkl')

    label_counter, labels = preprocessor.scan_labels()
    train_indexes, train_labels, test_indexes, test_labels = preprocessor.split_train_test(labels)

    dataset = DataPreprocessorDatasetOnline(preprocessor, tokenizer, train_indexes)

    print(dataset[0])


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.getLogger('transformers').setLevel('CRITICAL')
    logger = logging.getLogger(__file__)

    main()
