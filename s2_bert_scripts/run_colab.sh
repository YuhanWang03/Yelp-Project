#!/bin/bash

echo "Installing required dependencies..."
pip install -q transformers datasets accelerate evaluate scipy

PROJECT_ROOT="/content/drive/MyDrive/Yelp_Project"

# Define the parameter arrays to combine
# MODELS=("microsoft/deberta-v3-base")
MODELS=("microsoft/deberta-v3-base" "roberta-base" "distilbert-base-uncased")
MAX_LENGTHS=(128 256)
SEEDS=(42)

echo "Starting Colab full-parameter experiments..."

for model in "${MODELS[@]}"; do
    for max_len in "${MAX_LENGTHS[@]}"; do
        for seed in "${SEEDS[@]}"; do

            echo "================================================================"
            echo "Launching experiment: Model=$model | MaxLen=$max_len | Seed=$seed"
            echo "================================================================"

            # Call the Python engine and pass all parameters here
            python s2_bert_scripts/run_experiments.py \
                --project_root "$PROJECT_ROOT" \
                --model_name "$model" \
                --task_type "5_class" \
                --use_regression "False" \
                --learning_rate 5e-6 \
                --batch_size 4 \
                --num_epochs 3 \
                --max_length "$max_len" \
                --seed "$seed" \
                --grad_accum_steps 2 \
                --weight_decay 0.01 \
                --lr_scheduler_type "cosine" \
                --fp16 "False"

        done
    done
done
