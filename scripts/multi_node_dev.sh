#!/bin/bash


SERVER_IPS=(
192.168.2.14
192.168.2.15

)

WORK_DIR=/workspace2/Haiyang/Transformer/tft-multitrain2
PORT_NUM=30003
DEVICE_NUM=1

echo "=== $((${#SERVER_IPS[@]}))-Node Training Launch Script (Master Node) ==="
# Set environment variables

MASTER_IP=$(hostname -I | awk '{print $1}')
echo "Master IP: $MASTER_IP"

# Function 1: Set environment variables
set_env_vars() {
    local ip=$1
    
    # Network interface mapping for different IPs
    declare -A INTERFACE_MAP
    INTERFACE_MAP["192.168.2.10"]="enp7s0"
    INTERFACE_MAP["192.168.2.11"]="enp216s0"
    INTERFACE_MAP["192.168.2.12"]="enp129s0"
    INTERFACE_MAP["192.168.2.13"]="enp129s0"

    INTERFACE_MAP["192.168.2.14"]="ens3"
    INTERFACE_MAP["192.168.2.15"]="ens3"
    
    # Get interface name for current IP
    local interface=${INTERFACE_MAP[$ip]}
    if [ -z "$interface" ]; then
        interface="enp7s0"  # Default interface
    fi
    
    export NCCL_SOCKET_IFNAME=$interface
    export NCCL_TREE_THRESHOLD=0
    export NCCL_IB_TIMEOUT=23  # Increase timeout
    export NCCL_CONNECT_TIMEOUT=600  # 10 minute connection timeout
    export TORCH_DISTRIBUTED_DEBUG=OFF
    export NCCL_ASYNC_ERROR_HANDLING=1
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    
    echo "Set NCCL_SOCKET_IFNAME=$interface for IP $ip"
}

# Function 2: Activate virtual environment and change to work directory
setup_env() {
    echo "Executing: source /root/myvenv/bin/activate && cd $WORK_DIR"
    source /root/myvenv/bin/activate && cd $WORK_DIR
}

# Function 3: Execute training scripts (plan and torchrun)
run_training() {
    local ip=$1
    local node_rank=$2
    
    echo 'kill get_gpu_processes'
    echo "Executing: python scripts/kill_gpu_processes.py"
    python scripts/kill_gpu_processes.py
    
    echo "Executing: nohup torchrun --nproc_per_node=$DEVICE_NUM --nnodes=${#SERVER_IPS[@]} --node_rank=$node_rank --master_addr=$MASTER_IP --master_port=$PORT_NUM --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:$PORT_NUM scripts/train.py > $node_rank.log 2>&1 &"
    nohup torchrun \
        --nproc_per_node=$DEVICE_NUM \
        --nnodes=${#SERVER_IPS[@]} \
        --node_rank=$node_rank \
        --master_addr=$MASTER_IP \
        --master_port=$PORT_NUM \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_IP:$PORT_NUM \
        scripts/train.py > $node_rank.log 2>&1 &
}

# Process SERVER_IPS: start master training and create SLAVE_IPS array
SLAVE_IPS=()
for i in "${!SERVER_IPS[@]}"; do
    if [ "${SERVER_IPS[$i]}" = "$MASTER_IP" ]; then
        echo "Starting training on master node $MASTER_IP with rank 0..."
        (set_env_vars $MASTER_IP && setup_env && run_training $MASTER_IP 0) &
    else
        SLAVE_IPS+=("${SERVER_IPS[$i]}")
    fi
done

# SSH to each slave node and start training
for i in "${!SLAVE_IPS[@]}"; do
    IP=${SLAVE_IPS[$i]}
    SLAVE_RANK=$((i + 1))  # Slave ranks: 1, 2, 3...
    
    echo "Starting training on slave node $IP with rank $SLAVE_RANK..."
    ssh $IP "$(declare -f set_env_vars setup_env run_training); \
             export WORK_DIR=$WORK_DIR; export DEVICE_NUM=$DEVICE_NUM; export MASTER_IP=$MASTER_IP; \
             export PORT_NUM=$PORT_NUM; export SERVER_IPS=(${SERVER_IPS[@]}); \
             set_env_vars $IP && setup_env && run_training $IP $SLAVE_RANK" &
done

# Wait for all background processes
wait


