#!/bin/bash
# Interactive job script for HPCC cluster
# This script requests an interactive GPU node and runs training

echo "Requesting interactive GPU node..."
echo "This will give you a shell on a compute node with GPU access"

# Request interactive session with GPU
srun --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=8 \
     --gres=gpu:1 \
     --mem=32G \
     --time=48:00:00 \
     --partition=gpu \
     --pty bash -c '
     
# Load required modules
module purge
module load python/3.8
module load cuda/11.3
module load cudnn/8.2

# Activate virtual environment
# source ~/venv/bin/activate
# OR if using conda:
# conda activate your_env_name

# Navigate to project directory
cd ~/Codegen350

# Create logs directory
mkdir -p logs

echo "======================================"
echo "Interactive session started"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPUs available:"
nvidia-smi
echo "======================================"

# Run the training
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
echo "Training completed at $(date)"
echo "======================================"
'
