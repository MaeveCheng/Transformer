#!/bin/bash

SERVER_IPS=(
# 192.168.2.10
#192.168.2.11
# 192.168.2.12
# 192.168.2.13

#192.168.2.18
# 192.168.2.19
# 192.168.2.20
# 192.168.2.21

# 192.168.2.22
# 192.168.2.23
192.168.2.14
# 192.168.2.15
)

WORK_DIR=/workspace2/Maeve/Transformer_dev1
PORT_NUM=30003
DEVICE_NUM=8

# Check for signal parameter
SIGNAL=$1
if [ -z "$SIGNAL" ]; then
    echo "Usage: $0 {start|stop|restart}"
    echo "  start   - Start training without killing GPU processes"
    echo "  stop    - Only kill GPU processes on all nodes"
    echo "  restart - Kill GPU processes then start training"
    if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
        exit 1
    else
        return 1
    fi
fi

if [ "$SIGNAL" != "start" ] && [ "$SIGNAL" != "stop" ] && [ "$SIGNAL" != "restart" ]; then
    echo "Invalid signal: $SIGNAL"
    echo "Usage: $0 {start|stop|restart}"
    if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
        exit 1
    else
        return 1
    fi
fi


echo "=== $((${#SERVER_IPS[@]}))-Node Training Launch Script (Master Node) ==="
echo "Signal: $SIGNAL"
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
    INTERFACE_MAP["192.168.2.13"]="ens49"
    INTERFACE_MAP["192.168.2.18"]="enp129s0"
    INTERFACE_MAP["192.168.2.19"]="enp129s0"
    INTERFACE_MAP["192.168.2.20"]="enp129s0"
    INTERFACE_MAP["192.168.2.21"]="enp129s0"
    INTERFACE_MAP["192.168.2.22"]="enp7s0"
    INTERFACE_MAP["192.168.2.23"]="enp1s0"

    INTERFACE_MAP["192.168.2.14"]="ens3"
    INTERFACE_MAP["192.168.2.15"]="ens3"

    # Get interface name for current IP
    local interface=${INTERFACE_MAP[$ip]}
    if [ -z "$interface" ]; then
        interface="enp7s0"  # Default interface
    fi
    
    export NCCL_SOCKET_IFNAME=$interface
    export NCCL_TREE_THRESHOLD=0
    export NCCL_IB_TIMEOUT=40 # Increase timeout
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

# Function 3: Kill GPU processes
kill_gpu_processes() {
    echo 'kill get_gpu_processes'
    echo "Executing: python scripts/kill_gpu_processes.py"
    python scripts/kill_gpu_processes.py
}

# Function 4: Start torchrun training
start_torchrun() {
    local ip=$1
    local node_rank=$2
    
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

# Function 5: Start log monitor (only on master node)
start_log_monitor() {
    echo 'Starting log monitor on master node...'
    echo "Executing: nohup python scripts/log_monitor.py > log_monitor.log 2>&1 &"
    nohup python scripts/log_monitor.py > log_monitor.log 2>&1 &
    echo "Log monitor started, PID: $!"
}

# Function 6: Do start action
do_start() {
    local ip=$1
    local node_rank=$2
    start_torchrun $ip $node_rank
    # Start log monitor only on master node (rank 0)
    # if [ "$node_rank" = "0" ]; then
    #     start_log_monitor
    # fi
}

# Function 7: Do stop action
do_stop() {
    kill_gpu_processes
}

# Function 8: Do restart action
do_restart() {
    local ip=$1
    local node_rank=$2
    kill_gpu_processes
    start_torchrun $ip $node_rank
    # Start log monitor only on master node (rank 0)
    # if [ "$node_rank" = "0" ]; then
    #     start_log_monitor
    # fi
}

# Process SERVER_IPS: start master training and create SLAVE_IPS array
SLAVE_IPS=()
for i in "${!SERVER_IPS[@]}"; do
    if [ "${SERVER_IPS[$i]}" = "$MASTER_IP" ]; then
        echo "Executing $SIGNAL on master node $MASTER_IP with rank 0..."
        case $SIGNAL in
            start)
                (set_env_vars $MASTER_IP && setup_env && do_start $MASTER_IP 0) &
                ;;
            stop)
                (set_env_vars $MASTER_IP && setup_env && do_stop) &
                ;;
            restart)
                (set_env_vars $MASTER_IP && setup_env && do_restart $MASTER_IP 0) &
                ;;
        esac
    else
        SLAVE_IPS+=("${SERVER_IPS[$i]}")
    fi
done

# SSH to each slave node and execute action
for i in "${!SLAVE_IPS[@]}"; do
    IP=${SLAVE_IPS[$i]}
    SLAVE_RANK=$((i + 1))  # Slave ranks: 1, 2, 3...
    
    echo "Executing $SIGNAL on slave node $IP with rank $SLAVE_RANK..."
    case $SIGNAL in
        start)
            ssh $IP "$(declare -f set_env_vars setup_env kill_gpu_processes start_torchrun do_start); \
                     export WORK_DIR=$WORK_DIR; export DEVICE_NUM=$DEVICE_NUM; export MASTER_IP=$MASTER_IP; \
                     export PORT_NUM=$PORT_NUM; export SERVER_IPS=(${SERVER_IPS[@]}); \
                     set_env_vars $IP && setup_env && do_start $IP $SLAVE_RANK" &
            ;;
        stop)
            ssh $IP "$(declare -f set_env_vars setup_env kill_gpu_processes do_stop); \
                     export WORK_DIR=$WORK_DIR; export DEVICE_NUM=$DEVICE_NUM; export MASTER_IP=$MASTER_IP; \
                     export PORT_NUM=$PORT_NUM; export SERVER_IPS=(${SERVER_IPS[@]}); \
                     set_env_vars $IP && setup_env && do_stop" &
            ;;
        restart)
            ssh $IP "$(declare -f set_env_vars setup_env kill_gpu_processes start_torchrun do_restart); \
                     export WORK_DIR=$WORK_DIR; export DEVICE_NUM=$DEVICE_NUM; export MASTER_IP=$MASTER_IP; \
                     export PORT_NUM=$PORT_NUM; export SERVER_IPS=(${SERVER_IPS[@]}); \
                     set_env_vars $IP && setup_env && do_restart $IP $SLAVE_RANK" &
            ;;
    esac
done

# Wait for all background processes
wait


