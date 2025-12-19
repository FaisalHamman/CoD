#!/bin/bash
export PYTHONPATH=.

# Define subset sizes and set batch size
subset_sizes=(8 16 32 64 128 512)
batch_size=8
cfx_option="--cfx"
dataset_name="sentiment140"

# Define available GPUs and calculate number of GPU pairs
gpus=(0 1 2 3)
n_pairs=$(( ${#gpus[@]} / 2 ))

pids=()
i=0
for subset_size in "${subset_sizes[@]}"; do
  # Calculate pair index for the current job
  pair_index=$(( i % n_pairs ))
  gpu_lw=${gpus[$((2 * pair_index))]}
  gpu_kd=${gpus[$((2 * pair_index + 1))]}

  echo "Running experiments for subset size $subset_size..."

  echo "Running LW with subset size $subset_size and cfx option $cfx_option on GPU $gpu_lw..."
  CUDA_VISIBLE_DEVICES=$gpu_lw nohup python text-classification/ted_no_trainer.py \
    --model_name_or_path microsoft/deberta-v3-small \
    --teacher_model_name_or_path cliang1453/deberta-v3-base-sst2 \
    --model_type ted-deberta-v2 \
    --task_name $dataset_name \
    --per_device_train_batch_size $batch_size \
    --max_length 256 \
    --learning_rate 6e-5 --num_train_epochs 150 \
    --num_warmup_steps 0 \
    --teacher_filter_interval 2 \
    --kl_alpha 20 --mse_alpha 20 --filter_disabled \
    --output_dir ./ted_output/${dataset_name}/lwd --seed 42 --mixed_precision fp16 --save_best \
    --subset_size $subset_size $cfx_option --with_tracking &
  pid_lw=$!

  echo "Running KD with subset size $subset_size and cfx option $cfx_option on GPU $gpu_kd..."
  CUDA_VISIBLE_DEVICES=$gpu_kd nohup python text-classification/ted_no_trainer.py \
    --model_name_or_path microsoft/deberta-v3-small \
    --teacher_model_name_or_path cliang1453/deberta-v3-base-sst2 \
    --model_type ted-deberta-v2 \
    --task_name $dataset_name \
    --per_device_train_batch_size $batch_size \
    --max_length 256 \
    --learning_rate 6e-5 --num_train_epochs 150 \
    --num_warmup_steps 0 \
    --teacher_filter_interval 2 \
    --kl_alpha 20 --mse_alpha 0 --filter_disabled \
    --output_dir ./ted_output/${dataset_name}/kd --seed 42 --mixed_precision fp16 --save_best \
    --subset_size $subset_size $cfx_option --with_tracking &
  pid_kd=$!

  # Collect the PIDs for the current pair
  pids+=("$pid_lw" "$pid_kd")

  # If we've launched n_pairs experiments concurrently, then wait for them to finish
  if (( i % n_pairs == n_pairs - 1 )); then
    wait "${pids[@]}"
    pids=()
  fi

  (( i++ ))
done

# Wait for any remaining jobs before exiting.
if (( ${#pids[@]} > 0 )); then
  wait "${pids[@]}"
fi

echo "All tasks completed."
