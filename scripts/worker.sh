if [ "$MASTER_IP" == 0 ]
then
    MASTER_IP="$(hostname).platform-jobs"
fi

python ./modules/train.py --local_rank $LOCAL_RANK \ --dist_world_size $WORLD_SIZE \
                          --dist_backend nccl \
                          --dist_init_method "tcp://${MASTER_IP}:${MASTER_PORT}" "$@"