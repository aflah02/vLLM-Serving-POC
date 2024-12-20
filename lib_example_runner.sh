#!/bin/bash -l

#SBATCH -o SLURM_Logs/%x_%j_%A-%T.out
#SBATCH -e SLURM_Logs/%x_%j_%A-%T.err
#SBATCH -D ./
#SBATCH -J lib_example_runner

#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=224GB

#SBATCH --constraint="gpu"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a100:4

# Wall clock limit (max. is 24 hours):
#SBATCH --time=24:00:00

module load python-waterboa/2024.06 

# Activate Env -

source "/ptmp/afkhan/virtual_environments/vllm_env/bin/activate"

# Run Inference -

python lib_example_runner.py

echo "Inference Done, Exiting"

