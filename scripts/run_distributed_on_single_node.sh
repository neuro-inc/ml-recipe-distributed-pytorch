# Use this script in job's shell to run training on one node where more then one GPU is available

python ./modules/train.py --local_rank 0 --dist_backend nccl --dist_init_method tcp://127.0.0.1:9080 "$@"
