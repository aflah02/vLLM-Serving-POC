#!/bin/bash -l

#SBATCH -o SLURM_Logs/%x_%j.out
#SBATCH -e SLURM_Logs/%x_%j.err
#SBATCH -D ./
#SBATCH -J h-405b-alpha

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=224GB

#SBATCH --constraint="gpu-bw"
#SBATCH --partition="gpu-bw"
#SBATCH --gres=gpu:a100:4

# Wall clock limit (max. is 24 hours):
#SBATCH --time=24:00:00

module load python-waterboa/2024.06 

# Activate Env -

source "/ptmp/afkhan/virtual_environments/vllm_env/bin/activate"

# Define Variables -

model="/ptmp/afkhan/Models/pythia-14m"
tp_size=4
pp_size=1
prompt_path="/ptmp/afkhan/vLLM-Serving-POC/Data/Prompt_OHB_Chat_Alpha.txt"
save_path="/ptmp/afkhan/vLLM-Serving-POC/Data/Outputs_pythia-14m_OHB_Chat_Alpha_tp_4_pp_1.json"
queries_path="/ptmp/afkhan/vLLM-Serving-POC/Data/FAQ_en.csv"

# Run Inference -

echo "Running Inference"
echo "Model: $model"
echo "TP Size: $tp_size"
echo "PP Size: $pp_size"
echo "Save Path: $save_path"
echo "Queries Path: $queries_path"

python main.py --model $model --tp_size $tp_size --pp_size $pp_size --save_path $save_path --queries_path $queries_path --prompt_path $prompt_path
#python main.py --model "/ptmp/afkhan/Models/pythia-14m" --tp_size 4 --pp_size 1 --save_path "/ptmp/afkhan/vLLM-Serving-POC/Data/Outputs_pythia-14m_OHB_Chat_Alpha_tp_4_pp_1.json" --queries_path "/ptmp/afkhan/vLLM-Serving-POC/Data/FAQ_en.csv" --prompt_path "/ptmp/afkhan/vLLM-Serving-POC/Data/Prompt_OHB_Chat_Alpha.txt"

echo "Inference Done, Exiting"

