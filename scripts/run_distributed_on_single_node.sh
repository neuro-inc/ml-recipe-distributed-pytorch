# export OMP_NUM_THREADS (int)(multiprocessing.cpu_count() / nproc_per_node)

python ./modules/train.py --local_rank 0 --dist_backend nccl --dist_init_method tcp://127.0.0.1:9080 "$@"
