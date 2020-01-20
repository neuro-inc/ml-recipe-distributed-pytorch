import logging
import multiprocessing as mp

import numpy as np

logger = logging.getLogger(__file__)


# todo: not async
class AsyncDatasetProcessorIterator:
    def __init__(self, processor):
        self.manager = mp.Manager()
        self.pool_queue = self.manager.Queue(processor.buffer_size)
        self.pool = mp.Pool(processor.n_jobs)

        self.processor = processor

        self.jobs = []

        self.num_done_jobs = 0

    def _job_done(self, *args, **kwargs):
        self.num_done_jobs += 1

    @staticmethod
    def _worker_fun(dataset, idx, pool_queue):
        # print('HERE 6')
        chunks = dataset[idx]
        # print('HERE 7')
        for chunk in chunks:
            pool_queue.put(chunk)

    def __iter__(self):
        return self._generator()

    def _generator(self):
        idxs = range(len(self.processor.dataset))
        # print('HERE 1')
        if self.processor.shuffle:
            idxs = np.asarray(idxs)
            np.random.shuffle(idxs)
        # print('HERE 2')
        for idx in idxs:
            self.jobs.append(self.pool.apply_async(AsyncDatasetProcessorIterator._worker_fun,
                                                   (self.processor.dataset, idx, self.pool_queue),
                                                   callback=self._job_done))
        # print('HERE 3')
        batch = []
        while True:
            # print('HERE 4')
            chunk = self.pool_queue.get()
            # print('HERE 5')
            batch.append(chunk)
            if len(batch) == self.processor.batch_size:
                yield self.processor.process_batch(batch)
                batch = []

            if self.pool_queue.empty() and self.num_done_jobs == len(self.processor.dataset):
                break

        if len(batch):
            yield self.processor.process_batch(batch)

    def _close_jobs(self):
        self.pool.close()
        self.pool.join()

    def __del__(self):
        self._close_jobs()


class AsyncDatasetProcessor:
    def __init__(self, dataset, batch_size, *,
                 n_jobs=4,
                 collate_fun=None,
                 buffer_size=1024,
                 shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size

        self.collate_fun = collate_fun
        self.n_jobs = n_jobs

        self.buffer_size = buffer_size

        self.shuffle = shuffle

    def process_batch(self, batch):
        return self.collate_fun(batch) if self.collate_fun is not None else batch

    def __iter__(self):
        return iter(AsyncDatasetProcessorIterator(self))
