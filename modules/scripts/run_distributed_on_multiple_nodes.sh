# export NUM_GPUS=2
# export OMP_NUM_THREADS (int)(multiprocessing.cpu_count() / nproc_per_node)

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS "$@"