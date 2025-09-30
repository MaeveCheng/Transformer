#!/bin/bash
## 1. Set NCCL environment variables for better compatibility
#export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=^lo  # Exclude loopback interface
#export NCCL_IB_DISABLE=1       # Disable InfiniBand if not using it
#export NCCL_P2P_DISABLE=0      # Enable P2P for multi-GPU
#export NCCL_TREE_THRESHOLD=0   # Force tree algorithm
# 2. Set network timeout and retry settings
#export NCCL_TIMEOUT=600         # 10 minutes timeout
#export NCCL_CONNECT_TIMEOUT=600 # Connection timeout
#export NCCL_COMM_ABORT_ON_ERROR=0  # Don't abort on errors
# 3. TCP settings for better socket handling
#export NCCL_SOCKET_NTHREADS=4
#export NCCL_NSOCKS_PERTHREAD=4
# 4. Force TCP transport
#export NCCL_NET_PLUGIN=none

# Parse command line arguments
NPROC_PER_NODE=""
NNODES=""
NODE_RANK=""
MASTER_ADDR=""
MASTER_PORT=""
TRAINING_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --nproc_per_node=*)
            NPROC_PER_NODE="${1#*=}"
            shift
            ;;
        --nnodes=*)
            NNODES="${1#*=}"
            shift
            ;;
        --node_rank=*)
            NODE_RANK="${1#*=}"
            shift
            ;;
        --master_addr=*)
            MASTER_ADDR="${1#*=}"
            shift
            ;;
        --master_port=*)
            MASTER_PORT="${1#*=}"
            shift
            ;;
        *)
            # Collect remaining arguments for the training script
            TRAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# Check if nproc_per_node was provided
if [ -z "$NPROC_PER_NODE" ]; then
    echo "Error: --nproc_per_node parameter is required"
    echo "Usage: $0 --nproc_per_node=<number> [--nnodes=<number>] [--node_rank=<rank>] [--master_addr=<ip>] [--master_port=<port>] [training_args...]"
    echo "Example: $0 --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=192.168.2.11 --master_port=29500 --strategy ddp --devices 8"
    exit 1
fi

echo "Starting training pipeline..."
echo "Configuration:"
echo "  Number of processes per node: $NPROC_PER_NODE"
[ -n "$NNODES" ] && echo "  Number of nodes: $NNODES"
[ -n "$NODE_RANK" ] && echo "  Node rank: $NODE_RANK"
[ -n "$MASTER_ADDR" ] && echo "  Master address: $MASTER_ADDR"
[ -n "$MASTER_PORT" ] && echo "  Master port: $MASTER_PORT"
[ ${#TRAINING_ARGS[@]} -gt 0 ] && echo "  Training arguments: ${TRAINING_ARGS[*]}"

# Execute plan.py first
echo "Step 1: Running plan.py..."
python scripts/plan.py

# Check if plan.py executed successfully
if [ $? -ne 0 ]; then
    echo "Error: plan.py failed to execute"
    exit 1
fi

echo "plan.py completed successfully"

# Build torchrun command
echo "Step 2: Running distributed training with torchrun..."
TORCHRUN_CMD="torchrun --nproc_per_node=$NPROC_PER_NODE"

# Add optional parameters if provided
[ -n "$NNODES" ] && TORCHRUN_CMD="$TORCHRUN_CMD --nnodes=$NNODES"
[ -n "$NODE_RANK" ] && TORCHRUN_CMD="$TORCHRUN_CMD --node_rank=$NODE_RANK"
[ -n "$MASTER_ADDR" ] && TORCHRUN_CMD="$TORCHRUN_CMD --master_addr=$MASTER_ADDR"
[ -n "$MASTER_PORT" ] && TORCHRUN_CMD="$TORCHRUN_CMD --master_port=$MASTER_PORT"

# Add the training script and its arguments
TORCHRUN_CMD="$TORCHRUN_CMD scripts/train.py ${TRAINING_ARGS[*]}"

echo "Executing: $TORCHRUN_CMD"
$TORCHRUN_CMD

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
    exit 1
fi
