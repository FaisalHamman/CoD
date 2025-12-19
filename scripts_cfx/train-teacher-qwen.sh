#!/bin/bash
export PYTHONPATH=.

MODEL_NAME="Qwen/Qwen2.5-1.5B"
datasets=("sentiment140")

for dataset in "${datasets[@]}"; do
    output_dir="./teacher_models/${MODEL_NAME}/${dataset}/teacher_init"
    echo "Training on dataset: ${dataset} using distributed training across 4 GPUs"
    
    # Launch distributed training with torchrun
    torchrun \
        --nproc_per_node=4 \
        --master_port=$(shuf -i 25000-30000 -n 1) \
        text-classification/teacher_trainer_qwen.py \
        --model_name_or_path ${MODEL_NAME} \
        --task_name ${dataset} \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --max_length 256 \
        --learning_rate 2e-5 \
        --num_train_epochs 6 \
        --num_warmup_steps 0 \
        --output_dir ${output_dir} \
        --seed 42 \
        --mixed_precision fp16 \
        --save_best \
        --with_tracking \
        --pad_to_max_length \
        --gradient_checkpointing \
        --ddp_find_unused_parameters
    wait
done
