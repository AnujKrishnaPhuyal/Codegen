# Running on HPC Cluster

## Quick Start

### 1. Upload Your Code and Data

```bash
# From your local machine, upload to the cluster
scp -r Codegen350 username@hpc-cluster.edu:/home/username/
```

### 2. Prepare Your Environment on the Cluster

```bash
# SSH into the cluster
ssh username@hpc-cluster.edu

# Navigate to your project
cd ~/Codegen350

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers scikit-learn pandas numpy
```

### 3. Prepare Your Data Structure

Make sure your data directory looks like:

```
Codegen350/
├── data/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── run.py
├── model.py
├── early_stopping.py
└── submit_job.sh
```

### 4. Customize the SLURM Script

Edit `submit_job.sh` to match your cluster configuration:

```bash
# Edit the script
nano submit_job.sh
```

Important parameters to adjust:

- `--partition=gpu` - Change to your cluster's GPU partition name
- `--gres=gpu:1` - Number of GPUs (1, 2, 4, etc.)
- `--mem=32G` - Memory allocation
- `--time=48:00:00` - Maximum run time
- Module loads - Check available modules with `module avail`
- Data paths - Update paths to your actual data files
- Model path - Update model name if different

### 5. Submit the Job

```bash
# Make the script executable
chmod +x submit_job.sh

# Submit to SLURM
sbatch submit_job.sh
```

## Monitoring Your Job

```bash
# Check job status
squeue -u $USER

# View job details
scontrol show job <job_id>

# Check output logs (while running or after completion)
tail -f logs/train_<job_id>.out
tail -f logs/train_<job_id>.err

# Cancel a job if needed
scancel <job_id>
```

## Available Scripts

### 1. `submit_job.sh` - Single GPU Training

- Uses 1 GPU
- Recommended for initial testing
- Lower batch size

### 2. `submit_multi_gpu.sh` - Multi-GPU Training

- Uses 4 GPUs with DataParallel
- Higher batch size for faster training
- More memory required

## Common Issues and Solutions

### Issue 1: Module not found

```bash
# Check available modules
module avail python
module avail cuda

# Load the correct versions
module load python/3.8  # adjust version
module load cuda/11.3   # adjust version
```

### Issue 2: Out of memory

- Reduce `--train_batch_size` in the script
- Reduce `--block_size`
- Request more memory with `#SBATCH --mem=64G`

### Issue 3: Job pending

```bash
# Check why job is pending
squeue -u $USER --start

# Check partition limits
scontrol show partition gpu
```

### Issue 4: CUDA not available

- Verify GPU allocation: `echo $CUDA_VISIBLE_DEVICES`
- Check if GPU is accessible: `nvidia-smi`
- Ensure correct CUDA module is loaded

## Cluster-Specific Customizations

Different HPC clusters may have different configurations. Check with your system administrator or documentation for:

1. **Partition names**: `sinfo` to list available partitions
2. **GPU types**: Some clusters have different GPU types (V100, A100, etc.)
   ```bash
   #SBATCH --gres=gpu:v100:1  # Specify GPU type if needed
   ```
3. **QoS (Quality of Service)**: Some clusters require QoS specification
   ```bash
   #SBATCH --qos=normal
   ```
4. **Account**: Some clusters require account specification
   ```bash
   #SBATCH --account=your_account_name
   ```

## Advanced: Interactive Testing

For debugging, you can request an interactive session:

```bash
# Request interactive GPU node
srun --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=16G --time=2:00:00 --pty bash

# Once on the node, load modules and test
module load python cuda
python run.py --help
```

## Tips for Efficient HPC Usage

1. **Test with small data first**: Use a subset of your data to verify everything works
2. **Use array jobs**: If running multiple experiments
   ```bash
   #SBATCH --array=1-5  # Run 5 variations
   ```
3. **Checkpoint regularly**: Your code already saves best model, which is good
4. **Monitor resource usage**: Use `sacct -j <job_id> --format=JobID,Elapsed,MaxRSS,MaxVMSize`
5. **Email notifications**: Add to SLURM script:
   ```bash
   #SBATCH --mail-type=BEGIN,END,FAIL
   #SBATCH --mail-user=your.email@domain.com
   ```

## Expected Output

Your model checkpoints will be saved in:

```
saved_models/checkpoint-best-f1/model.bin
```

Logs will be in:

```
logs/train_<job_id>.out  # Standard output
logs/train_<job_id>.err  # Error messages
```
