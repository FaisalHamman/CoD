#!/bin/bash
export PYTHONPATH=.

# Define arrays
datasets=("amazon" "sst2" "cola" "imdb" "yelp" "sentiment140")
subset_sizes=(8 16 32 64 128 512)
teacher_models="microsoft/deberta-v3-base"
student_models="microsoft/deberta-v3-small"
gpus=(0 1 2 3)

batch_size=8
seed=0

# Counters for GPU assignment
teacher_count=0
student_count=0

# Launch teacher experiments
for dataset in "${datasets[@]}"; do
  for subset in "${subset_sizes[@]}"; do
    gpu_index=${gpus[$(( teacher_count % ${#gpus[@]} ))]}
    echo "Launching TEACHER experiment for dataset ${dataset} with subset ${subset} on GPU ${gpu_index}"
    CUDA_VISIBLE_DEVICES=${gpu_index} python text-classification/learn_filters_glue_no_trainer.py \
      --model_name_or_path "/export/fhamman/cfxkd/task-aware-distillation/teacher_models/${teacher_models}/${dataset}/teacher_init" \
      --model_type ted-deberta-v2 \
      --task_name ${dataset} \
      --per_device_train_batch_size ${batch_size} \
      --max_length 256 \
      --learning_rate 2e-5 --num_train_epochs 50 \
      --num_warmup_steps 0 \
      --filter_interval 1 \
      --output_dir "./ted_model/${teacher_models}/${dataset}/teacher_stage1" \
      --seed ${seed} --mixed_precision fp16 \
      --subset_size ${subset} \
      --data_type cfx \
      --use_slow_tokenizer &
    teacher_count=$(( teacher_count + 1 ))
  done
done

# Launch student experiments
for dataset in "${datasets[@]}"; do
  for subset in "${subset_sizes[@]}"; do
    gpu_index=${gpus[$(( student_count % ${#gpus[@]} ))]}
    echo "Launching STUDENT experiment for dataset ${dataset} with subset ${subset} on GPU ${gpu_index}"
    CUDA_VISIBLE_DEVICES=${gpu_index} python text-classification/learn_filters_glue_no_trainer.py \
      --model_name_or_path "${student_models}" \
      --model_type ted-deberta-v2 \
      --task_name ${dataset} \
      --per_device_train_batch_size ${batch_size} \
      --max_length 256 \
      --learning_rate 2e-5 --num_train_epochs 50 \
      --num_warmup_steps 0 \
      --filter_interval 1 --filter_output_dim 768 \
      --output_dir "./ted_model/${student_models}/${dataset}/student_stage1" \
      --seed ${seed} --mixed_precision fp16 \
      --subset_size ${subset} \
      --data_type cfx &
    student_count=$(( student_count + 1 ))
  done
done

wait
echo "All experiments launched."

