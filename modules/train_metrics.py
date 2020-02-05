import torch
import torch.multiprocessing as mp

from init import init_loss, init_model, init_datasets, init_collate_fun
from utils import get_logger, set_seed, show_params

from model.utils.parser import get_predictor_parser, get_model_parser, write_config_file, get_params
from model.dataset import RawPreprocessor
from model.trainer.callback import MAPCallback, AccuracyCallback, SaveBestCallback
from model.trainer.trainer import Trainer


def run_test(params):
    trainer = Trainer(model=params.model,
                      loss=params.loss,
                      collate_fun=params.collate_fun,

                      test_dataset=params.dataset,

                      device=params.device,

                      test_batch_size=params.batch_size,

                      n_jobs=params.n_jobs,

                      # apex_level=params.apex_level,
                      # apex_verbosity=params.apex_verbosity,
                      # apex_loss_scale=params.apex_loss_scale,
                      )

    callbacks = [MAPCallback(list(RawPreprocessor.labels2id.keys())),
                 AccuracyCallback()]

    trainer.test(-1, callbacks=callbacks)


def main(params, model_params) -> None:
    show_params(model_params, 'model')
    show_params(params, 'test')

    params.model, params.tokenizer = init_model(model_params, checkpoint=params.checkpoint,
                                                device=params.device)

    train_dataset, test_dataset, weights = init_datasets(params, tokenizer=params.tokenizer, clear=False)
    params.loss = init_loss(params, weights)

    params.collate_fun = init_collate_fun(params.tokenizer)

    logger.info('Train dataset validation..')
    params.dataset = train_dataset
    run_test(params)

    logger.info('Test dataset validation..')
    params.dataset = test_dataset
    run_test(params)


if __name__ == '__main__':
    (parser, model_parser), (params, model_params) = get_params((get_predictor_parser, get_model_parser))

    params.n_jobs = min(params.n_jobs, mp.cpu_count() // 2)
    params.device = torch.device('cuda') if torch.cuda.is_available() and params.gpu else torch.device('cpu')

    logger = get_logger()

    main(params, model_params)
