#!/bin/bash
#SBATCH --job-name=yolo-smoke-2g
#SBATCH --output=smoke-2g.%j.log
#SBATCH --error=smoke-2g.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

# load modules / activate environment as needed
# module load cuda python/3.11
module load python opencv
source ENV/bin/activate

# Optional NCCL debug / resilience
export TORCH_NCCL_ASYNC_HANDLING=1
export NCCL_DEBUG=INFO

# Determine per-process GPU count (nproc_per_node)
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-2}

# Threads / dataloader workers per process
THREADS_PER_PROC=$(( SLURM_CPUS_PER_TASK / NPROC_PER_NODE ))
if [ "$THREADS_PER_PROC" -lt 1 ]; then THREADS_PER_PROC=1; fi
export OMP_NUM_THREADS=$THREADS_PER_PROC
export OMP_THREAD_LIMIT=$THREADS_PER_PROC

NUM_WORKERS=$(( THREADS_PER_PROC - 1 ))
if [ "$NUM_WORKERS" -lt 1 ]; then NUM_WORKERS=1; fi

echo "HOST: $(hostname)"
echo "GPUS requested: ${NPROC_PER_NODE}"
echo "CPUS per task: ${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "DataLoader NUM_WORKERS(per-proc): ${NUM_WORKERS}"

# Run distributed training with 1 process per GPU
torchrun --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \
  multi_pipe.py \
  --distributed \
  --batch_size 4 \
  --num_workers ${NUM_WORKERS} > runtime_smoke.log 2>&1