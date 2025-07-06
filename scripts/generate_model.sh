#!/bin/bash

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PIFUHD_DIR="$PROJECT_ROOT/model/pifuhd"

# Change to PiFuHD directory
cd "$PIFUHD_DIR" || exit 1

# Get command line arguments
IMAGE_NAME=$1
RESOLUTION=${2:-256}

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Current directory: $(pwd)"
echo "Processing image: $IMAGE_NAME"
echo "Resolution: $RESOLUTION"

# Run PiFuHD
echo "Running PiFuHD..."
python -m apps.simple_test \
    -i "sample_images" \
    -o "results" \
    -c "checkpoints/pifuhd.pt" \
    -r "$RESOLUTION" \
    --use_rect

# Clean the generated mesh
echo "Cleaning generated mesh..."
python apps/clean_mesh.py -f "results/pifuhd_final/recon"

echo "Process completed!" 