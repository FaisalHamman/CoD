#!/bin/bash
export PYTHONPATH=.

# Define datasets, subset sizes and batch size
# datasets=("sst2" "yelp" "sentiment140")

datasets=("sst2") #"" "sentiment140" "cola")
subset_sizes=(8 16 32 64 128 512)
batch_size=8

# Define available GPUs (4 GPUs)
gpus=(0 1 2 3)

for dataset in "${datasets[@]}"; do
  echo "Running experiments for dataset $dataset..."
  for subset_size in "${subset_sizes[@]}"; do
    echo "  Running experiments for subset size $subset_size..."
    
    # Initialize an array to store PIDs for parallel processes
    pids=()
    
    # Run KD_clean experiment on GPU 0
    echo "    Running KD_clean on GPU ${gpus[0]}..."
    CUDA_VISIBLE_DEVICES=${gpus[0]} nohup python text-classification/ted_no_trainer.py \
      --model_name_or_path microsoft/deberta-v3-small \
      --teacher_model_name_or_path cliang1453/deberta-v3-base-sst2 \
      --model_type ted-deberta-v2 \
      --task_name $dataset \
      --per_device_train_batch_size $batch_size \
      --max_length 256 \
      --learning_rate 6e-5 --num_train_epochs 75 \
      --num_warmup_steps 0 --teacher_filter_interval 2 \
      --kl_alpha 20 --mse_alpha 0 --filter_disabled \
      --output_dir ./ted_output/${dataset}/kd --seed 42 \
      --mixed_precision fp16 --save_best --subset_size $subset_size \
      --data_type clean  --with_tracking & 
    pids+=($!)
    
    # Run LWD_clean experiment on GPU 1
    echo "    Running LWD_clean on GPU ${gpus[1]}..."
    CUDA_VISIBLE_DEVICES=${gpus[1]} nohup python text-classification/ted_no_trainer.py \
      --model_name_or_path microsoft/deberta-v3-small \
      --teacher_model_name_or_path cliang1453/deberta-v3-base-sst2 \
      --model_type ted-deberta-v2 \
      --task_name $dataset \
      --per_device_train_batch_size $batch_size \
      --max_length 256 \
      --learning_rate 6e-5 --num_train_epochs 75 \
      --num_warmup_steps 0 --teacher_filter_interval 2 \
      --kl_alpha 20 --mse_alpha 20 --filter_disabled \
      --output_dir ./ted_output/${dataset}/lwd --seed 42 \
      --mixed_precision fp16 --save_best --subset_size $subset_size \
      --data_type clean  --with_tracking & 
    pids+=($!)
    
    # Run KD_cfx experiment on GPU 2
    echo "    Running KD_cfx on GPU ${gpus[2]}..."
    CUDA_VISIBLE_DEVICES=${gpus[2]} nohup python text-classification/ted_no_trainer.py \
      --model_name_or_path microsoft/deberta-v3-small \
      --teacher_model_name_or_path cliang1453/deberta-v3-base-sst2 \
      --model_type ted-deberta-v2 \
      --task_name $dataset \
      --per_device_train_batch_size $batch_size \
      --max_length 256 \
      --learning_rate 6e-5 --num_train_epochs 75 \
      --num_warmup_steps 0 --teacher_filter_interval 2 \
      --kl_alpha 20 --mse_alpha 0 --filter_disabled \
      --output_dir ./ted_output/${dataset}/kd --seed 42 \
      --mixed_precision fp16 --save_best --subset_size $subset_size \
      --data_type cfx  --with_tracking &
    pids+=($!)
    
    # Run LWD_cfx experiment on GPU 3
    echo "    Running LWD_cfx on GPU ${gpus[3]}..."
    CUDA_VISIBLE_DEVICES=${gpus[3]} nohup python text-classification/ted_no_trainer.py \
      --model_name_or_path microsoft/deberta-v3-small \
      --teacher_model_name_or_path cliang1453/deberta-v3-base-sst2 \
      --model_type ted-deberta-v2 \
      --task_name $dataset \
      --per_device_train_batch_size $batch_size \
      --max_length 256 \
      --learning_rate 6e-5 --num_train_epochs 75 \
      --num_warmup_steps 0 --teacher_filter_interval 2 \
      --kl_alpha 20 --mse_alpha 20 --filter_disabled \
      --output_dir ./ted_output/${dataset}/lwd --seed 42 \
      --mixed_precision fp16 --save_best --subset_size $subset_size \
      --data_type cfx  --with_tracking &
    pids+=($!)
    
    # Wait for all 4 experiments to complete for the current subset size
    for pid in "${pids[@]}"; do
      wait $pid
    done
    echo "  Completed experiments for subset size $subset_size."
  done
  echo "Completed experiments for dataset $dataset."
done

echo "All tasks completed."
