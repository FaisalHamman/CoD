#!/bin/bash
export PYTHONPATH=.

# Define arrays
SEEDS=(0 1 2 3 4)
datasets=("amazon" "sst2" "cola" "imdb" "yelp")

subset_sizes=(8 16 32 64 128 512)
batch_size=8
gpus=(0 1 2 3)

# Initialize GPU counters (each GPU may run up to 5 experiments concurrently)
declare -A gpu_count
for gpu in "${gpus[@]}"; do
  gpu_count[$gpu]=0
done

# Map PID to GPU
declare -A pid2gpu

# Build a list of experiments.
# Each entry is a pipe-delimited list of parameters in order:
# 1 model, 2 teacher_model, 3 model_type, 4 task_name, 5 batch_size, 6 max_length,
# 7 learning_rate, 8 num_train_epochs, 9 num_warmup_steps, 10 teacher_filter_interval,
# 11 kl_alpha, 12 mse_alpha, 13 filter_disabled, 14 filter_output_dim, 15 output_dir,
# 16 mixed_precision, 17 save_best, 18 subset_size, 19 data_type, 20 with_tracking, 21 seed

commands=()
for dataset in "${datasets[@]}"; do
  for subset_size in "${subset_sizes[@]}"; do
    for seed in "${SEEDS[@]}"; do
      # TED_clean experiment
      commands+=("./ted_model/microsoft/deberta-v3-small/${dataset}/student_stage1_subset_${subset_size}_clean_seed0|./ted_model/microsoft/deberta-v3-base/${dataset}/teacher_stage1_subset_${subset_size}_clean_seed0|qwen|${dataset}|${batch_size}|256|6e-5|40|0|2|20|20|768|./ted_output/microsoft/deberta-v3-small/${dataset}/seed${seed}/ted|fp16|save_best|${subset_size}|clean|with_tracking|${seed}")
      
      # TED_cfx experiment
      commands+=("./ted_model/microsoft/deberta-v3-small/${dataset}/student_stage1_subset_${subset_size}_cfx_seed0|./ted_model/microsoft/deberta-v3-base/${dataset}/teacher_stage1_subset_${subset_size}_clean_seed0|qwen|${dataset}|${batch_size}|256|6e-5|40|0|2|20|20|768|./ted_output/microsoft/deberta-v3-small/${dataset}/seed${seed}/ted|fp16|save_best|${subset_size}|cfx|with_tracking|${seed}")
    done
  done
done

total=${#commands[@]}
echo "Total experiments: $total"

# Loop over each experiment command and schedule it when a GPU slot is available.
for cmd in "${commands[@]}"; do
  available_gpu=""
  # Wait until at least one GPU has fewer than 5 active jobs.
  while true; do
    for gpu in "${gpus[@]}"; do
      if [ ${gpu_count[$gpu]} -lt 8 ]; then
         available_gpu=$gpu
         break
      fi
    done
    if [ -n "$available_gpu" ]; then
      break
    fi
    # Instead of wait -n, check all running PIDs to see if a job finished.
    finished_pid=""
    for pid in "${!pid2gpu[@]}"; do
      if ! kill -0 $pid 2>/dev/null; then
        finished_pid=$pid
        break
      fi
    done
    if [ -z "$finished_pid" ]; then
      sleep 1
      continue
    fi
    finished_gpu=${pid2gpu[$finished_pid]}
    gpu_count[$finished_gpu]=$(( gpu_count[$finished_gpu] - 1 ))
    unset pid2gpu[$finished_pid]
  done

  # Parse the command string.
  IFS='|' read -ra parts <<< "$cmd"
  model_name_or_path=${parts[0]}
  teacher_model_name_or_path=${parts[1]}
  model_type=${parts[2]}
  task_name=${parts[3]}
  per_device_train_batch_size=${parts[4]}
  max_length=${parts[5]}
  learning_rate=${parts[6]}
  num_train_epochs=${parts[7]}
  num_warmup_steps=${parts[8]}
  teacher_filter_interval=${parts[9]}
  kl_alpha=${parts[10]}
  mse_alpha=${parts[11]}
  filter_output_dim=${parts[12]}
  output_dir=${parts[13]}
  mixed_precision=${parts[14]}
  save_best=${parts[15]}
  subset_size=${parts[16]}
  data_type=${parts[17]}
  with_tracking=${parts[18]}
  seed=${parts[19]}

  # Build and launch the command.
  run_cmd="CUDA_VISIBLE_DEVICES=$available_gpu nohup python text-classification/ted_no_trainer.py \
    --model_name_or_path \"$model_name_or_path\" \
    --teacher_model_name_or_path \"$teacher_model_name_or_path\" \
    --model_type \"$model_type\" \
    --task_name \"$task_name\" \
    --per_device_train_batch_size \"$per_device_train_batch_size\" \
    --max_length \"$max_length\" \
    --learning_rate \"$learning_rate\" --num_train_epochs \"$num_train_epochs\" \
    --num_warmup_steps \"$num_warmup_steps\" --teacher_filter_interval \"$teacher_filter_interval\" \
    --kl_alpha \"$kl_alpha\" --mse_alpha \"$mse_alpha\" --filter_output_dim \"$filter_output_dim\"  \
    --output_dir \"$output_dir\" \
    --mixed_precision \"$mixed_precision\" --$save_best --subset_size \"$subset_size\" \
    --data_type \"$data_type\" --with_tracking --seed \"$seed\" &"
  eval $run_cmd
  pid=$!
  pid2gpu[$pid]=$available_gpu
  gpu_count[$available_gpu]=$(( gpu_count[$available_gpu] + 1 ))
done

# Wait for all remaining jobs.
wait
echo "All tasks completed."




