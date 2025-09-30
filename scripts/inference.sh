#!/bin/bash

# Inference runner script for Transformer model
# This script sets up the Python path and runs inference

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Set Python path to include the project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Use the virtual environment Python
PYTHON_CMD="/home/max/Projects/python/ml-env/bin/python"

# Check if Python is available
if [ ! -f "$PYTHON_CMD" ]; then
    print_error "Python not found at $PYTHON_CMD"
    exit 1
fi

# Print Python version
print_info "Using Python: $($PYTHON_CMD --version)"

# Check if required modules exist
print_info "Checking required modules..."
$PYTHON_CMD -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}'); import lightning.inference" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "lightning.inference module not found, checking setup..."
    print_info "Project structure:"
    ls -la "${PROJECT_ROOT}/lightning/"
fi

# Print usage if no arguments
if [ $# -eq 0 ]; then
    print_info "Usage examples:"
    echo ""
    echo "  Single file prediction:"
    echo "    ./scripts/inference.sh predict --checkpoint model.ckpt --input data.parquet"
    echo ""
    echo "  Directory processing:"
    echo "    ./scripts/inference.sh predict --checkpoint model.ckpt --input ./test/ --output ./predictions/"
    echo ""
    echo "  Batch processing:"
    echo "    ./scripts/inference.sh batch --checkpoint model.ckpt --input-dir ./data/"
    echo ""
    echo "  Your command:"
    echo "    ./scripts/inference.sh predict --checkpoint logs/default/checkpoints/step_step=002700.ckpt --input ../Transformer2/train_data_bn/ --output ./predicts/"
    echo ""
    exit 0
fi

# Run the inference script with all arguments
print_info "Running inference with arguments: $@"
print_info "Project root: ${PROJECT_ROOT}"
print_info "PYTHONPATH: ${PYTHONPATH}"

# Create parent directory for output if specified
if [[ "$@" == *"--output"* ]]; then
    OUTPUT_PATH=$(echo "$@" | sed -n 's/.*--output \([^ ]*\).*/\1/p')
    if [ ! -z "$OUTPUT_PATH" ]; then
        OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
        mkdir -p "$OUTPUT_DIR"
        print_info "Output will be saved to: $OUTPUT_PATH"
    fi
fi

# Create output directory for batch mode
if [[ "$@" == *"--output-dir"* ]]; then
    OUTPUT_DIR=$(echo "$@" | sed -n 's/.*--output-dir \([^ ]*\).*/\1/p')
    if [ ! -z "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
        print_info "Created output directory: $OUTPUT_DIR"
    fi
fi

# Change to project root to ensure imports work
cd "${PROJECT_ROOT}"

# Run the inference script
$PYTHON_CMD scripts/inference.py "$@"

# Check exit status
if [ $? -eq 0 ]; then
    print_info "Inference completed successfully!"
else
    print_error "Inference failed. Check the error messages above."
    exit 1
fi