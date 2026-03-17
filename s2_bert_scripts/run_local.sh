#!/bin/bash
# run_local.sh

# ==========================================
# 1. Basic configuration
# ==========================================
# Update this to your actual local project path (use forward slashes).
PROJECT_ROOT="D:/CSE4601_Text Mining/Yelp_Project"

# ==========================================
# 2. Define the hyperparameter combinations to test
# ==========================================
MODELS=("distilbert-base-uncased" "roberta-base")
TASK_TYPES=("binary" "3_class")
BATCH_SIZES=(2 4)

echo "Starting local grid search experiments..."

# ==========================================
# 3. Run experiments with nested loops
# ==========================================
for model in "${MODELS[@]}"; do
    for task in "${TASK_TYPES[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do

            echo "================================================================"
            echo "Launching experiment: Model=$model | Task=$task | BatchSize=$bs"
            echo "================================================================"

            # Call the Python script with command-line arguments
            python s2_bert_scripts/run_experiments.py \
                --project_root "$PROJECT_ROOT" \
                --model_name "$model" \
                --task_type "$task" \
                --use_regression "False" \
                --batch_size "$bs" \
                --num_epochs 1  # Run only 1 epoch locally to verify the pipeline

        done
    done
done

echo "All local experiment schedules have finished."
