#!/bin/bash


SEEDS=(0 1 2 3 4)

datasets=("amazon" "sst2" "cola" "imdb" "yelp" "sentiment140")

MODEL="gpt-4o"

# Define the sample sizes
SAMPLE_SIZES=(8 16 32 64 128 512)


# Loop through each dataset, sample size, and seed, then run the script
for ds in "${datasets[@]}"; do
    for SAMPLE_SIZE in "${SAMPLE_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Running counterfactual generation for dataset: $ds, sample size: $SAMPLE_SIZE samples, seed: $seed..."
            python ./main.py \
                --return_dataset "cfx" \
                --dataset_name "$ds" \
                --model "$MODEL" \
                --subset_size "$SAMPLE_SIZE" \
                --output_dir "./cfx-dataset" \
                --seed "$seed"
        done
    done
done

echo "All tasks completed."


