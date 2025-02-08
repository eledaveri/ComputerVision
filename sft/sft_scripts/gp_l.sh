#!/bin/bash

#SBATCH --job-name=sft
#SBATCH --mail-user=user@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1                                 # Total number of nodes requested
#SBATCH --ntasks-per-node=8                       # Total number of task requested
#SBATCH --cpus-per-task=8                        # Total number of cores requested
#SBATCH --mem=1000G
#SBATCH -t 72:00:00                          # Time limit (hh:mm:ss)
#SBATCH --gpus-per-node=8   
#SBATCH --output=sft_%A_%a.out
#SBATCH --output=sft_%A_%a.err
#SBATCH --partition=gpu
export PYTHONPATH=src:$PYTHONPATH
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
DATA_JSON="YOUR_JSON_DATA_PATH"
IMAGE_FOLDER="./"
OUTPUT_FOLDER="../train_ckpt/gp_l_sft"

# gp-l
LR=1e-6
EPOCH=1
# gp-vl
# LR=5e-7

# virl-l
# LR=1e-6

# virl-vl
# LR=5e-7
# EPOCH=5

deepspeed src/training/train.py \
    --deepspeed sft_scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path $DATA_JSON \
    --image_folder $IMAGE_FOLDER \
    --disable_flash_attn2 True \
    --lora_enable False \
    --tune_img_projector True \
    --freeze_vision_tower False \
    --freeze_llm False \
    --bf16 True \
    --output_dir $OUTPUT_FOLDER \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --projector_lr ${LR} \
    --vision_lr ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to none \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 20 \
    --dataloader_num_workers 4 \
    --save_only_model True