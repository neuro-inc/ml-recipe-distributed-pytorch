# Change WORLD_SIZE variable to change the number of jobs used during distributed training
# Change PRESET variable to hardware specification which is used by jobs
WORLD_SIZE=2
PRESET='gpu-small'

MASTER_PORT=9080

SCRIPT_NAME='worker.sh'
CONFIG_NAME='test_bert.cfg'

echo "Running the master job..."

make dist DIST_WAIT_START=yes \
          PRESET=$PRESET \
          LOCAL_RANK=0 \
          WORLD_SIZE=$WORLD_SIZE \
          MASTER_IP=0 \
          MASTER_PORT=$MASTER_PORT \
          SCRIPT_NAME=$SCRIPT_NAME \
          CONFIG_NAME=$CONFIG_NAME \
          RUN=master

MASTER_IP=$(neuro status dist-distributed-pytorch-master | awk '/Internal Hostname:/ {print $3}')

echo "Running worker jobs..."

for ((i=1;i<$WORLD_SIZE;i++))
do
    make dist DIST_WAIT_START=no \
              PRESET=$PRESET \
              LOCAL_RANK=$i \
              WORLD_SIZE=$WORLD_SIZE \
              MASTER_IP=$MASTER_IP \
              MASTER_PORT=$MASTER_PORT \
              SCRIPT_NAME=$SCRIPT_NAME \
              CONFIG_NAME=$CONFIG_NAME \
              RUN=worker-$i
done

echo "All jobs were initialized."
echo "Streaming logs of the job dist-distributed-pytorch-master"
neuro exec --no-key-check -T dist-distributed-pytorch-master "tail -f -n 1000000 /output" || echo -e "Stopped streaming logs.\nUse 'neuro logs <job>' to see full logs."
