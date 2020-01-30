WORLD_SIZE=2
LOCAL_RANK=0

MASTER_PORT=9080
MASTER_IP=0

PRESET='gpu-small'

SCRIPT_NAME='worker.sh'
CONFIG_NAME='test_bert.cfg'

echo "Running the master job..."

make dist DIST_WAIT_START=yes \
          PRESET=$PRESET \
          LOCAL_RANK=$LOCAL_RANK \
          WORLD_SIZE=$WORLD_SIZE \
          MASTER_IP=$MASTER_IP \
          MASTER_PORT=$MASTER_PORT \
          SCRIPT_NAME=$SCRIPT_NAME \
          CONFIG_NAME=$CONFIG_NAME \
          RUN=master

MASTER_IP=$(neuro status dist-qa-competition-master | awk '/Internal Hostname:/ {print $3}')

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
echo "Streaming logs of the job dist-qa-competition-master"
neuro exec --no-key-check -T dist-qa-competition-master "tail -f -n 1000000 /output" || echo -e "Stopped streaming logs.\nUse 'neuro logs <job>' to see full logs."
