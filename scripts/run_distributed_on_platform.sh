WORLD_SIZE=2
LOCAL_RANK=0

MASTER_PORT=9080
MASTER_IP=0

PRESET='gpu-small'

SCRIPT_NAME='worker.sh'
CONFIG_NAME='test_bert.cfg'

make dist PRESET=$PRESET \
          LOCAL_RANK=$LOCAL_RANK \
          WORLD_SIZE=$WORLD_SIZE \
          MASTER_IP=$MASTER_IP \
          MASTER_PORT=$MASTER_PORT \
          SCRIPT_NAME=$SCRIPT_NAME \
          CONFIG_NAME=$CONFIG_NAME \
          RUN=master &

# idk: sometimes it fails to run worker jobs
sleep 10

MASTER_IP=$(neuro status dist-qa-competition-master | awk '/Internal Hostname:/ {print $3}')

for ((i=1;i<$WORLD_SIZE;i++))
do
    make dist PRESET=$PRESET \
              LOCAL_RANK=$i \
              WORLD_SIZE=$WORLD_SIZE \
              MASTER_IP=$MASTER_IP \
              MASTER_PORT=$MASTER_PORT \
              SCRIPT_NAME=$SCRIPT_NAME \
              CONFIG_NAME=$CONFIG_NAME \
              RUN=worker-$i &> /dev/null &
done
