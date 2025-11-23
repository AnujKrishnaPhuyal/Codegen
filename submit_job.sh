#!/bin/bash
#SBATCH --job-name=model_training          # Job name
#SBATCH --output=logs/train_%j.out         # Standard output log (%j expands to jobId)
#SBATCH --error=logs/train_%j.err          # Standard error log
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks-per-node=1                # Number of tasks per node
#SBATCH --cpus-per-task=8                  # Number of CPU cores per task
#SBATCH --gres=gpu:1                       # Number of GPUs (adjust as needed)
#SBATCH --mem=32G                          # Memory per node (adjust as needed)
#SBATCH --time=48:00:00                    # Time limit (hrs:min:sec)
#SBATCH --partition=gpu                    # Partition name (adjust to your cluster)

# Load required modules (adjust based on your cluster)
module purge
module load python/3.8
module load cuda/11.3
module load cudnn/8.2

# Activate your virtual environment (if using one)
# source ~/venv/bin/activate
# OR if using conda:
# conda activate your_env_name

# Navigate to your project directory
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Print some information about the job
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Run the training script
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

echo "Job finished at $(date)"
