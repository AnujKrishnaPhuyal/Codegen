#!/bin/bash
#SBATCH --job-name=model_training_multigpu
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4                       # Request 4 GPUs
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu

# Load required modules
module purge
module load python/3.8
module load cuda/11.3
module load cudnn/8.2

# Activate virtual environment
# source ~/venv/bin/activate

cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"

# For multi-GPU training, the script already uses DataParallel
# Just make sure all GPUs are visible
export CUDA_VISIBLE_DEVICES=0,1,2,3

python run.py \
    --output_dir=./saved_models \
    --model_name_or_path=Salesforce/codegen-350M-mono \
    --train_data_file=./data/train.jsonl \
    --eval_data_file=./data/valid.jsonl \
    --test_data_file=./data/test.jsonl \
    --block_size=512 \
    --do_train \
    --do_test \
    --train_batch_size=64 \
    --eval_batch_size=128 \
    --learning_rate=2e-5 \
    --num_train_epochs=10 \
    --seed=42 \
    --contrast

echo "Job finished at $(date)"
