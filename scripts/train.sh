#!/bin/bash

# Change to the project directory
cd "$(dirname "$0")/.." || exit 1

# Parse command line arguments
NPROC_PER_NODE=""

for arg in "$@"; do
    case $arg in
        --nproc_per_node=*)
            NPROC_PER_NODE="${arg#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 --nproc_per_node=<number>"
            echo "Example: $0 --nproc_per_node=4"
            exit 1
            ;;
    esac
done

# Check if nproc_per_node was provided
if [ -z "$NPROC_PER_NODE" ]; then
    echo "Error: --nproc_per_node parameter is required"
    echo "Usage: $0 --nproc_per_node=<number>"
    echo "Example: $0 --nproc_per_node=4"
    exit 1
fi

echo "Starting training pipeline..."
echo "Number of processes per node: $NPROC_PER_NODE"

# Execute plan.py first (optional - skip if not found)
if [ -f "scripts/plan.py" ]; then
    echo "Step 1: Running plan.py..."
    /home/max/Projects/python/ml-env/bin/python scripts/plan.py
    
    # Check if plan.py executed successfully
    if [ $? -ne 0 ]; then
        echo "Warning: plan.py failed to execute, continuing anyway..."
    else
        echo "plan.py completed successfully"
    fi
else
    echo "Step 1: Skipping plan.py (file not found)"
fi

# Execute torchrun with train.py
echo "Step 2: Running distributed training with torchrun..."
export PYTHONPATH=/home/max/Projects/python/Transformer:$PYTHONPATH
source /home/max/Projects/python/ml-env/bin/activate && torchrun --nproc_per_node=$NPROC_PER_NODE scripts/train.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
    exit 1
fi