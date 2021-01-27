# Change WORLD_SIZE variable to change the number of jobs used during distributed training
WORLD_SIZE=2

echo "Running the master job..."

neuro-flow run distributed_training --param world_size ${WORLD_SIZE} --param name distributed-pytorch-master

MASTER_IP=$(neuro status distributed-pytorch-master | awk '/Internal Hostname / {print $3}' | head -1)

echo "Running worker jobs..."

for ((i=1;i<$WORLD_SIZE;i++))
do
    neuro-flow run distributed_training --param world_size ${WORLD_SIZE} --param name "distributed-pytorch-worker-${i}" --param master_ip "$MASTER_IP" --param local_rank $i 
done

echo "All jobs were initialized."
echo "Streaming logs of the job dist-distributed-pytorch-master"
neuro logs distributed-pytorch-master 
