# Running Interactive Jobs on HPCC Cluster

## Method 1: Fully Automated Interactive Job

This method requests an interactive node and automatically runs your training.

### Step 1: Customize the script

```bash
nano run_interactive.sh
```

**Adjust these parameters:**

- Line 16: `--partition=gpu` (change to your cluster's partition)
- Line 20-22: Module names
- Line 25-27: Virtual environment activation
- Line 37-48: Training arguments

### Step 2: Make it executable and run

```bash
chmod +x run_interactive.sh
./run_interactive.sh
```

**What happens:**

1. Requests an interactive GPU node
2. Loads modules automatically
3. Runs your training
4. You can see output in real-time
5. Can press Ctrl+C to stop if needed

---

## Method 2: Manual Interactive Session (More Control)

This gives you a shell on a GPU node where you run commands manually.

### Step 1: Start interactive session

```bash
chmod +x start_interactive.sh
./start_interactive.sh
```

OR directly use srun:

```bash
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=32G --time=48:00:00 --partition=gpu --pty bash
```

### Step 2: Once you get the node prompt, run commands:

```bash
# Load modules
module load python/3.8 cuda/11.3 cudnn/8.2

# Activate environment
cd ~/Codegen350
source venv/bin/activate

# Check GPU
nvidia-smi

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
```

### Step 3: Monitor in real-time

- You'll see all output directly in your terminal
- Can stop with Ctrl+C
- Session ends when training completes or time limit reached

### Step 4: Exit

```bash
exit  # Exits the interactive node
```

---

## Method 3: Interactive with Screen/Tmux (Recommended for Long Jobs)

This allows you to disconnect and reconnect without losing your session.

### Step 1: Start screen before requesting node

```bash
screen -S training  # Creates a named screen session
```

### Step 2: Request interactive node

```bash
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=32G --time=48:00:00 --partition=gpu --pty bash
```

### Step 3: Run your training

```bash
cd ~/Codegen350
source venv/bin/activate
module load python cuda

python run.py [your arguments]
```

### Step 4: Detach and reattach

```bash
# Detach from screen: Press Ctrl+A then D
# Your job keeps running!

# Reattach later:
screen -r training

# List all screens:
screen -ls
```

---

## Quick Comparison

| Method             | Best For              | Pros                             | Cons                          |
| ------------------ | --------------------- | -------------------------------- | ----------------------------- |
| Batch (sbatch)     | Long unattended runs  | Runs in background, job queuing  | Can't see real-time output    |
| Interactive Auto   | Testing, debugging    | See output live, automated setup | Must stay connected           |
| Interactive Manual | Full control          | Complete flexibility             | More manual steps             |
| Screen/Tmux        | Long interactive runs | Can disconnect/reconnect         | Requires learning screen/tmux |

---

## Interactive Job Parameters

Adjust these based on your needs:

```bash
--nodes=1                    # Number of nodes (usually 1)
--ntasks-per-node=1          # Tasks per node (1 for single process)
--cpus-per-task=8            # CPU cores (increase for data loading)
--gres=gpu:1                 # Number of GPUs (1, 2, 4, etc.)
--mem=32G                    # Memory (16G, 32G, 64G, etc.)
--time=48:00:00              # Max time (HH:MM:SS)
--partition=gpu              # Partition name (check with sinfo)
```

### For different GPU types:

```bash
--gres=gpu:v100:1            # Request specific GPU type
--gres=gpu:a100:2            # Request 2 A100 GPUs
```

---

## Troubleshooting

### Issue: "srun: error: Unable to allocate resources"

**Solution:** Resources busy, try:

```bash
# Check queue
squeue

# Request fewer resources
srun --gres=gpu:1 --mem=16G --time=4:00:00 --partition=gpu --pty bash

# Check available partitions
sinfo -o "%20P %5a %.10l %16F %N"
```

### Issue: Session times out

**Solution:** Use screen/tmux or reduce time request

### Issue: Can't see GPU

**Solution:**

```bash
# Check if GPU allocated
echo $CUDA_VISIBLE_DEVICES

# Check GPU status
nvidia-smi

# Ensure correct modules loaded
module list
```

### Issue: Connection dropped

**Solution:**

- Always use screen/tmux for long runs
- Or use batch jobs (sbatch) instead

---

## Testing Before Full Training

Always test with a short interactive session first:

```bash
# Request 30-minute session
srun --gres=gpu:1 --mem=16G --time=0:30:00 --partition=gpu --pty bash

# Test with 1 epoch
cd ~/Codegen350
source venv/bin/activate
module load python cuda

python run.py \
    --output_dir=./test_models \
    --model_name_or_path=Salesforce/codegen-350M-mono \
    --train_data_file=./data/train.jsonl \
    --eval_data_file=./data/valid.jsonl \
    --block_size=512 \
    --do_train \
    --train_batch_size=8 \
    --eval_batch_size=16 \
    --num_train_epochs=1 \
    --seed=42

# If this works, exit and run full training
exit
```

---

## Recommended Workflow

1. **First time:** Use Method 2 (manual) to test everything works
2. **Short runs:** Use Method 1 (automated interactive)
3. **Long runs:** Use batch jobs (sbatch) or Method 3 (screen/tmux)
4. **Development:** Use short interactive sessions for debugging

---

## Commands Cheat Sheet

```bash
# Request interactive session
srun --gres=gpu:1 --mem=32G --time=8:00:00 --partition=gpu --pty bash

# Inside session - check resources
nvidia-smi                   # Check GPU
free -h                      # Check memory
lscpu                        # Check CPUs
hostname                     # Check node name

# Run training in background (within interactive session)
nohup python run.py [args] > train.log 2>&1 &

# Monitor background job
tail -f train.log
jobs
fg %1                        # Bring to foreground
```
