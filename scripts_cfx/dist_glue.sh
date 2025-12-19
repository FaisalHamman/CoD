#!/bin/bash
export PYTHONPATH=.

# Define subset sizes and counterfactual options
subset_sizes=(8 16 32 64 128)
# Set batch size
batch_size=8 #32
cfx_options=("" "--cfx")

# Define available GPUs
gpus=(0 1 2 3)

for subset_size in "${subset_sizes[@]}"; do
  echo "Running experiments for subset size $subset_size..."

  # Initialize an array to store PIDs for parallel processes
  pids=()

  for i in "${!cfx_options[@]}"; do
    cfx_option="${cfx_options[$i]}"

    # Run LW experiment on the assigned GPU
    gpu_id=${gpus[$((2 * i))]}  # Assign GPU for LW
    echo "Running LW with subset size $subset_size and cfx option $cfx_option on GPU $gpu_id..."
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python text-classification/ted_glue_no_trainer.py \
      --model_name_or_path microsoft/deberta-v3-xsmall --teacher_model_name_or_path cliang1453/deberta-v3-base-sst2 \
      --model_type ted-deberta-v2 \
      --task_name sst2 \
      --per_device_train_batch_size $batch_size \
      --max_length 256 \
      --learning_rate 4e-5 --num_train_epochs 150 \
      --num_warmup_steps 0 \
      --kl_alpha 20 --mse_alpha 20 --filter_disabled \
      --output_dir ./ted_output/sst2/lwd --seed 42 --mixed_precision fp16 --save_best \
      --subset_size $subset_size $cfx_option --with_tracking &
      # --measure_energy &

    # Store the PID of the LW process
    pids+=($!)


    

    # Run KD experiment on the assigned GPU
    gpu_id=${gpus[$((2 * i + 1))]}  # Assign GPU for KD
    echo "Running KD with subset size $subset_size and cfx option $cfx_option on GPU $gpu_id..."
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python text-classification/ted_glue_no_trainer.py \
      --model_name_or_path microsoft/deberta-v3-xsmall --teacher_model_name_or_path cliang1453/deberta-v3-base-sst2 \
      --model_type ted-deberta-v2 \
      --task_name sst2 \
      --per_device_train_batch_size $batch_size \
      --max_length 256 \
      --learning_rate 4e-5 --num_train_epochs 150 \
      --num_warmup_steps 0 \
      --kl_alpha 20 --mse_alpha 0 --filter_disabled \
      --output_dir ./ted_output/sst2/kd --seed 42 --mixed_precision fp16 --save_best \
      --subset_size $subset_size $cfx_option --with_tracking &
      # --measure_energy &

    # Store the PID of the KD process
    pids+=($!)
  done

  # Wait for all processes for the current subset size to complete
  for pid in "${pids[@]}"; do
    wait $pid
  done

  echo "Completed experiments for subset size $subset_size."
done

echo "All tasks completed."