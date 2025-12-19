#!/bin/bash
export PYTHONPATH=.

# Define arrays
SEEDS=(0 1 2 3 4)
datasets=("amazon" "sst2" "cola" "imdb" "yelp" "sentiment140")

subset_sizes=(8) # 16 32 64 128 512)
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
# model, teacher_model, model_type, task_name, batch_size, max_length, learning_rate, num_train_epochs,
# num_warmup_steps, teacher_filter_interval, kl_alpha, mse_alpha, filter_disabled, output_dir,
# mixed_precision, save_best, subset_size, data_type, with_tracking, seed
commands=()
for dataset in "${datasets[@]}"; do
  for subset_size in "${subset_sizes[@]}"; do
    for seed in "${SEEDS[@]}"; do
      # KD_clean experiment
      commands+=("microsoft/deberta-v3-xsmall|teacher_models/microsoft/deberta-v3-base/${dataset}/teacher_init|ted-deberta-v2|${dataset}|${batch_size}|256|6e-5|100|0|1|20|0|filter_disabled|./ted_output/microsoft/deberta-v3-xsmall/${dataset}/seed${seed}/kd|fp16|save_best|${subset_size}|clean|with_tracking|${seed}")
      # LWD_clean experiment
      commands+=("microsoft/deberta-v3-xsmall|teacher_models/microsoft/deberta-v3-base/${dataset}/teacher_init|ted-deberta-v2|${dataset}|${batch_size}|256|6e-5|100|0|1|20|20|filter_disabled|./ted_output/microsoft/deberta-v3-xsmall/${dataset}/seed${seed}/lwd|fp16|save_best|${subset_size}|clean|with_tracking|${seed}")
      # KD_cfx experiment
      commands+=("microsoft/deberta-v3-xsmall|teacher_models/microsoft/deberta-v3-base/${dataset}/teacher_init|ted-deberta-v2|${dataset}|${batch_size}|256|6e-5|100|0|1|20|0|filter_disabled|./ted_output/microsoft/deberta-v3-xsmall/${dataset}/seed${seed}/kd|fp16|save_best|${subset_size}|cfx|with_tracking|${seed}")
      # LWD_cfx experiment
      commands+=("microsoft/deberta-v3-xsmall|teacher_models/microsoft/deberta-v3-base/${dataset}/teacher_init|ted-deberta-v2|${dataset}|${batch_size}|256|6e-5|100|0|1|20|20|filter_disabled|./ted_output/microsoft/deberta-v3-xsmall/${dataset}/seed${seed}/lwd|fp16|save_best|${subset_size}|cfx|with_tracking|${seed}")
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
      if [ ${gpu_count[$gpu]} -lt 12 ]; then
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
  filter_disabled=${parts[12]}
  output_dir=${parts[13]}
  mixed_precision=${parts[14]}
  save_best=${parts[15]}
  subset_size=${parts[16]}
  data_type=${parts[17]}
  with_tracking=${parts[18]}
  seed=${parts[19]}

  # Build and launch the command.
  run_cmd="CUDA_VISIBLE_DEVICES=$available_gpu nohup python text-classification/ted_no_trainer_qwen.py \
    --model_name_or_path $model_name_or_path \
    --teacher_model_name_or_path $teacher_model_name_or_path \
    --model_type $model_type \
    --task_name $task_name \
    --per_device_train_batch_size $per_device_train_batch_size \
    --max_length $max_length \
    --learning_rate $learning_rate --num_train_epochs $num_train_epochs \
    --num_warmup_steps $num_warmup_steps --teacher_filter_interval $teacher_filter_interval \
    --kl_alpha $kl_alpha --mse_alpha $mse_alpha --$filter_disabled \
    --output_dir $output_dir \
    --mixed_precision $mixed_precision --$save_best --subset_size $subset_size \
    --data_type $data_type  --with_tracking --seed $seed &"
  eval $run_cmd
  pid=$!
  pid2gpu[$pid]=$available_gpu
  gpu_count[$available_gpu]=$(( gpu_count[$available_gpu] + 1 ))
done

# Wait for all remaining jobs.
wait
echo "All tasks completed."




