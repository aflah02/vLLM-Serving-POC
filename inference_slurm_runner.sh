#!/bin/bash -l

#SBATCH -o SLURM_Logs/job.out.%j
#SBATCH -e SLURM_Logs/job.err.%j
#SBATCH -D ./
#SBATCH -J llm_inference

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=224GB

#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4

# Wall clock limit (max. is 24 hours):
#SBATCH --time=24:00:00

module load python-waterboa/2024.06 

# Activate Env -

source "/ptmp/afkhan/virtual_environments/vllm_env/bin/activate"

# Define Variables -

model = "/ptmp/afkhan/Models/Llama-3.1-70B-Instruct"
tp_size = 4
pp_size = 1
save_path = ""
queries_path = ""

# Run Inference -

echo "Running Inference"
echo "Model: $model"
echo "TP Size: $tp_size"
echo "PP Size: $pp_size"
echo "Save Path: $save_path"
echo "Queries Path: $queries_path"

python main.py --model $model --tp_size $tp_size --pp_size $pp_size --save_path $save_path --queries_path $queries_path

echo "Inference Done, Exiting"

