# export LOCAL_RANK=0
# export MASTER_IP=127.0.0.1
# export MASTER_PORT=9080

python ./modules/train.py --local_rank $LOCAL_RANK --dist_backend nccl --dist_init_method "tcp://${MASTER_IP}:${MASTER_PORT}" "$@"
