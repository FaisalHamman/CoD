#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=.


# MODEL_NAME="Qwen/Qwen2.5-1.5B"
MODEL_NAME="Qwen/Qwen2.5-7B"
# MODEL_NAME="microsoft/deberta-v3-base"
datasets=("amazon" "sst2" "cola" "imdb" "yelp" "sentiment140")

max_jobs=4
job_count=0

for i in "${!datasets[@]}"; do
    dataset=${datasets[i]}
    gpu=$(( i % max_jobs ))
    output_dir="./teacher_models/${MODEL_NAME}/${dataset}/teacher_init"

    echo "Training on dataset: ${dataset} using GPU: ${gpu}"
    
    CUDA_VISIBLE_DEVICES=${gpu} python text-classification/teacher_trainer.py\
      --model_name_or_path ${MODEL_NAME} \
      --task_name ${dataset} \
      --per_device_train_batch_size 8 \
      --max_length 256 \
      --learning_rate 2e-5 --num_train_epochs 6 \
      --num_warmup_steps 0 \
      --output_dir ${output_dir} \
      --seed 42 --mixed_precision fp16 --save_best \
      --with_tracking &

    job_count=$(( job_count + 1 ))
    if [ ${job_count} -ge ${max_jobs} ]; then
        wait
        job_count=0
    fi
done
wait