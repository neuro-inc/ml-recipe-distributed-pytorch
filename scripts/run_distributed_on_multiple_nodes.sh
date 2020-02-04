 # Script was used to debug distributed training
 # Use this script in job's shell to run training on multiple nodes. The number of GPU must be the same on all nodes.
 # Don't forget to change exported bellow variables

 export LOCAL_RANK=0
 export MASTER_IP=job-6ccd8dd5-649d-4720-b82b-ea7c4c11a463.platform-jobs
 export MASTER_PORT=9080
 export WORLD_SIZE=2

python ./modules/train.py --local_rank $LOCAL_RANK --dist_world_size $WORLD_SIZE --dist_backend nccl --dist_init_method "tcp://${MASTER_IP}:${MASTER_PORT}" "$@"
