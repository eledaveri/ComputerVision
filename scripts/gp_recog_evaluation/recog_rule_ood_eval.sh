#!/bin/bash

#SBATCH --job-name=EvalGP
#SBATCH --mail-user=user@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1                                 # Total number of nodes requested
#SBATCH --ntasks-per-node=1                       # Total number of task requested
#SBATCH --cpus-per-task=8                        # Total number of cores requested
#SBATCH --mem=512G
#SBATCH -t 72:00:00                          # Time limit (hh:mm:ss)
#SBATCH --gpus-per-node=1   
#SBATCH --output=./slurm_logs/eval_%A_%a.out
#SBATCH --output=./slurm_logs/eval_%A_%a.err
#SBATCH --partition=gpu

VITER=1
# disable verification to evaluate vision
ENABLE=False
# enable rule-ood eval
OOD=True
# choose rule: face card as 11,12,13
FACE10=False
# specify target: 24
TARGET=24
# specify card color: black
COLOR=black

NUM_TRAJ=234
CKPT_NAME="HuggingFaceTB/SmolVLM-256M-Instruct"
OUTPUT_FOLDER="logs/gp_vl_recog_rule_ood_verify_${VITER}_target_${TARGET}"
PORT=$((RANDOM % 10000 + 1000))

accelerate launch \
    --config_file scripts/config_zero2_1gpu.yaml --main_process_port ${PORT} \
    -m evaluation.launcher -f evaluation/configs/llama_gp_vl.yaml \
    --model_path=${CKPT_NAME} \
    --output_dir=${OUTPUT_FOLDER}/gp_vl_recog_rule_ood.jsonl \
    --prompt_config.enable_verification=${ENABLE} \
    --env_config.target_points=${TARGET} \
    --env_config.verify_iter=${VITER} \
    --env_config.treat_face_cards_as_10=${FACE10} \
    --env_config.face_cards_color=${COLOR} \
    --env_config.ood=${OOD} \
    --num_traj=${NUM_TRAJ}