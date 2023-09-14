#!/bin/bash
# export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1,5,3,4
cd /mnt/lustre/sjtu/home/zym22/huggingface/Llama-X/src

# python -m debugpy --listen 5678 --wait-for-client train.py \
deepspeed train_librispeech.py \
    --model_name_or_path /data/LM/llama-7b-hf \
    --data_path /mnt/lustre/sjtu/shared/data/asr/rawdata \
    --output_dir /mnt/lustre/sjtu/home/zym22/models/llama \
    --num_train_epochs 2 \
    --model_max_length 2048 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed /mnt/lustre/sjtu/home/zym22/huggingface/Llama-X/src/configs/deepspeed_config.json \
    --fp16 True \
    --cache_dir /mnt/lustre/sjtu/home/zym22/huggingface/Llama-X/data/cache \