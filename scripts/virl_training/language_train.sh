#!/bin/bash

#SBATCH --job-name=TrainVI
#SBATCH --mail-user=user@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1                                 # Total number of nodes requested
#SBATCH --ntasks-per-node=8                      # Total number of task requested
#SBATCH --cpus-per-task=8                        # Total number of cores requested
#SBATCH --mem=1000G
#SBATCH --time=96:00:00                     # Time limit (hh:mm:ss)
#SBATCH --gpus-per-node=8   
#SBATCH --output=./slurm_logs/train_%A_%a.out
#SBATCH --output=./slurm_logs/train_%A_%a.err
#SBATCH --partition=gpu

LR=1e-6
save_every=1
save_model=False # disable running saving. one checkpoint ~30GB

CKPT_NAME="HuggingFaceTB/SmolVLM-500M-Instruct" # official init model: tianzhechu/GP-L-Init
PORT=$((RANDOM % 10000 + 1000))

# download from our huggingface dataset repo tianzhechu/SFTvsRL_Data
ROUTE_INFO="SFTvsRL_DATA_new/VIRL_routes/nyc_1k_routes/route_infos.json" # .json
GPS_TO_PANO="SFTvsRL_DATA_new/VIRL_routes/nyc_1k_routes/gps_pano_mapping.pkl" # .pkl
STREETVIEWS="SFTvsRL_DATA_new/VIRL_routes/nyc_1k_routes/street_views" # folder of images

DS_SKIP_CUDA_CHECK=1 TOKENIZERS_PARALLELISM=false \
    accelerate launch \
    --config_file scripts/config_zero2_8gpu.yaml \
    --main_process_port ${PORT} -m rl.launcher \
    -f rl/configs/llama_virl_language.yaml \
    --output_dir=train_ckpt/virl_l/ \
    --optimizer_config.init_lr=${LR} \
    --optimizer_config.lr_max_steps=20 \
    --prompt_config.enable_verification=True \
    --num_updates=15 \
    --run_name=virl_language_training \
    --num_steps=256 \
    --model_path=${CKPT_NAME} \
    --save_ckpt=${save_model} \
    --save_every=${save_every} \
    --env_config.route_info_path=${ROUTE_INFO} \
    --env_config.platform_cfg.OFFLINE.PANORAMA_DIR=${STREETVIEWS} \
    --env_config.platform_cfg.OFFLINE.GPS_TO_PANO_PATH=${GPS_TO_PANO} 
