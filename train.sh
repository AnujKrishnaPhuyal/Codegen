#!/bin/bash
# Simple training script for interactive session
# Run this after you've already gotten an interactive GPU node

# Load required modules (adjust based on your cluster)
module purge
module load python/3.8
module load cuda/11.3
module load cudnn/8.2

# Activate virtual environment (uncomment the one you're using)
# source ~/venv/bin/activate
# OR
# conda activate your_env_name

# Navigate to project directory
cd ~/Codegen350

# Create logs and output directories
mkdir -p logs
mkdir -p saved_models

# Check GPU availability
# echo "======================================"
# echo "Checking GPU availability..."
# nvidia-smi
# echo "======================================"

# Run training
python run.py \
    --output_dir=./saved_models \
    --model_name_or_path=Salesforce/codegen-350M-mono \
    --train_data_file=./data/train.jsonl \
    --eval_data_file=./data/valid.jsonl \
    --test_data_file=./data/test.jsonl \
    --block_size=512 \
    --do_train \
    --do_test \
    --train_batch_size=16 \
    --eval_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=10 \
    --seed=42 \
    --contrast

echo "======================================"
echo "Training completed!"
echo "======================================"
