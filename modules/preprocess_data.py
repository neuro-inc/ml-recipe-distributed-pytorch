import logging

import configargparse
from model.dataset import RawDataPreprocessor
from transformers import BertTokenizer


def get_parser() -> configargparse.ArgumentParser:
    parser = configargparse.ArgumentParser(description='Midi-generator training script.')

    parser.add_argument('-c', '--config_file', required=False, is_config_file=True, help='Config file path.')

    parser.add_argument('--in_path', type=str, required=True, help='')
    parser.add_argument('--out_path', type=str, required=True, help='')

    parser.add_argument('--max_seq_len', type=int, default=384, help='')
    parser.add_argument('--max_question_len', type=int, default=64, help='')
    parser.add_argument('--doc_stride', type=int, default=128, help='')

    parser.add_argument('--limit_data', type=int, default=0, help='')

    parser.add_argument('--num_threads', type=int, default=2, help='')

    return parser


def show_params(params: configargparse.Namespace) -> None:
    logger.info('Input parameters:')
    for k in sorted(params.__dict__.keys()):
        logger.info(f'\t{k}: {getattr(params, k)}')


def main() -> None:
    params = get_parser().parse_args()
    show_params(params)

    bert_model = 'bert-base-uncased'
    do_lower_case = 'uncased' in bert_model

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    data_preprocessor = RawDataPreprocessor(in_path=params.in_path,
                                            out_path=params.out_path,
                                            tokenizer=tokenizer,
                                            max_seq_len=params.max_seq_len,
                                            max_question_len=params.max_question_len,
                                            doc_stride=params.doc_stride,
                                            limit_data=params.limit_data,
                                            num_threads=params.num_threads
                                            )

    data_preprocessor()


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.getLogger('transformers').setLevel('CRITICAL')
    logger = logging.getLogger(__file__)

    main()
