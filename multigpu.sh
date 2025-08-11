#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=0-1:00
#SBATCH --output=%N-%j.out

module load python opencv
source ENV/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_THREAD_LIMIT=$SLURM_CPUS_PER_TASK
export TORCH_NCCL_ASYNC_HANDLING=1
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=$(( ( RANDOM % 1000 ) + 28000 ))  # Choose a randomized high port in [28000–28999]
export MASTER_PORT

srun python -u 'multi Pipe.py' \
    --distributed \
    --batch_size   32 \
    --num_workers  4 \
    --init_method  tcp://$MASTER_ADDR:$MASTER_PORT \
    --world_size   $SLURM_NTASKS \
    --rank         $SLURM_PROCID \
    --local_rank   $SLURM_LOCALID
