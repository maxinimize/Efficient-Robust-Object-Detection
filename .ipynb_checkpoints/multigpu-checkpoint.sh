#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # 1 task per node; torchrun will spawn per-GPU procs
#SBATCH --gres=gpu:h100:4           # request 4 H100 GPUs
#SBATCH --cpus-per-task=16          # 16 CPUs per task (4 per GPU)
#SBATCH --mem=50G                   # 50GB RAM
#SBATCH --time=0-1:00               # 1 hour
#SBATCH --output=%N-%j.out          # output file

set -euo pipefail
module load python opencv
source ENV/bin/activate

# Enable NCCL async error handling and debug output for distributed training
export TORCH_NCCL_ASYNC_HANDLING=1
export NCCL_DEBUG=INFO

# Determine number of processes per node (GPUs)
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  NPROC_PER_NODE=$SLURM_GPUS_ON_NODE
else
  # Try to detect visible GPUs (requires nvidia-smi)
  NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || true)
  if [ -n "$NGPU" ] && [ "$NGPU" -gt 0 ]; then
    NPROC_PER_NODE=$NGPU
  else
    # final fallback
    NPROC_PER_NODE=1
  fi
fi

# Allocate CPU threads per DDP process (avoid oversubscription)
THREADS_PER_PROC=$(( SLURM_CPUS_PER_TASK / NPROC_PER_NODE ))
if [ "$THREADS_PER_PROC" -lt 1 ]; then
  THREADS_PER_PROC=1
fi

# Set OpenMP thread limits for each process
export OMP_NUM_THREADS=$THREADS_PER_PROC
export OMP_THREAD_LIMIT=$THREADS_PER_PROC

# Use (threads per proc - 1) dataloader workers per process, min 1
NUM_WORKERS=$(( THREADS_PER_PROC - 1 ))
if [ "$NUM_WORKERS" -lt 1 ]; then
  NUM_WORKERS=1
fi

# Debug lines to capture what the job sees
echo "===== debug env ====="
echo "HOST: $(hostname)"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"
echo "SLURM_LOCALID=${SLURM_LOCALID:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Detected NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "OMP_NUM_THREADS(per-proc)=${OMP_NUM_THREADS}"
echo "DataLoader NUM_WORKERS(per-proc)=${NUM_WORKERS}"
echo "---------------------"

# Run torchrun (note: script filename has no spaces)
torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \
  multi_pipe.py \
  --distributed \
  --batch_size 32 \
  --num_workers ${NUM_WORKERS}
