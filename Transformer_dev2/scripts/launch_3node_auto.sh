#!/bin/bash
# Auto-detect GPU count and launch appropriately

if [ $# -ne 1 ]; then
    echo "Usage: $0 <node_rank>"
    echo "  node_rank: 0 for xiaoniu2, 1 for xiaoniu3, 2 for xiaoniu4"
    exit 1
fi

NODE_RANK=$1
MASTER_ADDR="xiaoniu2"
MASTER_PORT=30002

echo "=== 3-Node Auto-GPU Launch ==="
echo "Node rank: $NODE_RANK"
echo "Hostname: $(hostname)"

# Detect number of GPUs
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
if [ "$GPU_COUNT" -eq 0 ]; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
fi

echo "Detected GPUs: $GPU_COUNT"

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "ERROR: No GPUs detected!"
    exit 1
fi

# Set environment variables
export NODE_RANK=$NODE_RANK
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_TREE_THRESHOLD=0
export NCCL_P2P_LEVEL=LOC
export NCCL_CROSS_NIC=0

# Kill old processes
pkill -f torchrun || true
sleep 2

# Log file
LOG_FILE="train_node${NODE_RANK}_gpu${GPU_COUNT}.log"

echo "Launching with $GPU_COUNT GPUs..."
echo "IMPORTANT: All nodes must use the same number of GPUs!"
echo ""

# Calculate total world size based on GPUs per node
# Assuming all nodes have the same GPU count as the first node
WORLD_SIZE=$((GPU_COUNT * 3))

echo "Starting torchrun with:"
echo "  GPUs per node: $GPU_COUNT"
echo "  Total processes: $WORLD_SIZE"
echo ""

nohup torchrun \
    --nproc_per_node=$GPU_COUNT \
    --nnodes=3 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py > $LOG_FILE 2>&1 &

PID=$!
echo "Started with PID: $PID"
echo "Log: $LOG_FILE"
echo ""
echo "Monitor with: tail -f $LOG_FILE"