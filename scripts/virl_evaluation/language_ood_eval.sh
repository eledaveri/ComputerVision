#!/bin/bash

#SBATCH --job-name=EvalVI
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

VITER=2

# enable verification
ENABLE=True
# use relative action space, different with training
ABS=False
NUM_TRAJ=48
CKPT_NAME="HuggingFaceTB/SmolVLM-500M-Instruct"
OUTPUT_FOLDER="logs/virl_l_ood_verify_${VITER}"
PORT=$((RANDOM % 10000 + 2000))

# download from our huggingface dataset repo tianzhechu/SFTvsRL_Data
ROUTE_INFO="SFTvsRL_DATA_new/VIRL_routes/nyc_1k_routes/route_infos.json" # .json
GPS_TO_PANO="SFTvsRL_DATA_new/VIRL_routes/nyc_1k_routes/gps_pano_mapping.pkl" # .pkl
STREETVIEWS="SFTvsRL_DATA_new/VIRL_routes/nyc_1k_routes/street_views" # folder of images

DS_SKIP_CUDA_CHECK=1 accelerate launch \
    --config_file scripts/config_zero2_1gpu.yaml --main_process_port ${PORT}  \
    -m evaluation.launcher -f evaluation/configs/llama_virl_language.yaml \
    --model_path=${CKPT_NAME} \
    --output_dir=${OUTPUT_FOLDER}/virl_l_ood.jsonl \
    --env_config.route_info_path=${ROUTE_INFO} \
    --env_config.platform_cfg.OFFLINE.PANORAMA_DIR=${STREETVIEWS} \
    --env_config.platform_cfg.OFFLINE.GPS_TO_PANO_PATH=${GPS_TO_PANO} \
    --prompt_config.enable_verification=${ENABLE} \
    --env_config.verify_iter=${VITER} \
    --env_config.absolute_action=${ABS} \
    --num_traj=${NUM_TRAJ}
