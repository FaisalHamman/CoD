#!/bin/bash

# Define the seeds
SEEDS=(0 1 2 3 4)

# Define the datasets
datasets=("amazon" "sst2" "cola" "imdb" "yelp" "sentiment140")

# Define the sample sizes
SAMPLE_SIZES=(8 16 32 64 128 512)

# Loop through each dataset, sample size, and seed, then run the script
for ds in "${datasets[@]}"; do
    for SAMPLE_SIZE in "${SAMPLE_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Running dataset generation for dataset: $ds, sample size: $SAMPLE_SIZE samples, seed: $seed..."
            python ./main.py \
                --return_dataset "clean" \
                --dataset_name "$ds" \
                --output_dir "./dataset" \
                --subset_size "$SAMPLE_SIZE" \
                --seed "$seed"
        done
    done
done

echo "All tasks completed."