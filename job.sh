#!/bin/bash
#SBATCH --job-name=sdf-models-training-test
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=80
#SBATCH --time=00-00:10:00
#SBATCH --output=slurm-j.out
#SBATCH --error=slurm-j.err
#SBATCH --mail-user=hliu9@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Load modules
source /etc/profile.d/modules.sh
module load openmpi/4.1.5-gcc-11.3.1-2clspqh

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_PARTITION"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per CPU: $SLURM_MEM_PER_CPU"
echo "Time limit: $SLURM_TIME_LIMIT"

# Your commands here
cd /home/hliu9/sdf-models/
python src/main.py

echo "Job completed successfully!"
