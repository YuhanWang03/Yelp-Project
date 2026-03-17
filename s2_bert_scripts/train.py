import os
import gc
import torch
import shutil
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# Import custom modules
from config import BASE_CONFIG
from data_loader import load_and_preprocess_data, tokenize_data
from utils import (
    setup_experiment_paths,
    get_compute_metrics_fn,
    save_evaluation_artifacts
)


def run_single_experiment(custom_config=None, project_root_path=None):
    """
    Core function for running a single training experiment.

    Args:
        custom_config: Dictionary used to override default values in BASE_CONFIG
            (for example, when sweeping over multiple learning rates).
        project_root_path: Path to the project root directory, used to support
            both local and Colab-style environments.
    """
    # 1. Merge configurations
    config = BASE_CONFIG.copy()
    if custom_config:
        config.update(custom_config)

    set_seed(config["seed"])
    project_root = Path(project_root_path) if project_root_path else Path.cwd()

    print("=" * 60)
    print(
        f"Starting experiment | Model: {config['model_name']} | "
        f"Task: {config['task_type']} | Regression: {config['use_regression']}"
    )
    print("=" * 60)

    try:
        # 2. Set up paths and auto-incremented run ID
        paths = setup_experiment_paths(config, project_root)
        model_run_dir, output_dir, best_model_dir, logging_dir, results_run_dir, run_id = paths

        train_path = project_root / "data" / "processed" / "train_data.csv"
        val_path = project_root / "data" / "processed" / "val_data.csv"

        # 3. Load and preprocess data
        train_dataset, val_dataset, num_labels = load_and_preprocess_data(
            train_path=train_path,
            val_path=val_path,
            task_type=config["task_type"],
            use_regression=config["use_regression"]
        )

        # 4. Initialize tokenizer and tokenize the datasets
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        tokenized_train, tokenized_val = tokenize_data(
            train_dataset, val_dataset, tokenizer, config["max_length"]
        )

        # 5. Load the model
        print(f"Loading model architecture: {config['model_name']} ...")
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"],
            num_labels=num_labels
        )

        # 6. Configure TrainingArguments
        metric_name = "mse" if config["use_regression"] else "accuracy"
        greater_is_better = False if config["use_regression"] else True

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["grad_accum_steps"],
            num_train_epochs=config["num_epochs"],
            weight_decay=config["weight_decay"],
            warmup_ratio=config["warmup_ratio"],
            lr_scheduler_type=config["lr_scheduler_type"],
            fp16=config["fp16"],
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            greater_is_better=greater_is_better,
            logging_dir=str(logging_dir),
            logging_steps=100,
            report_to="none",  # Disable default wandb/tensorboard reporting for a cleaner setup
        )

        # 7. Initialize the Trainer
        compute_metrics = get_compute_metrics_fn(config)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=compute_metrics,
        )

        # ==========================================
        # 8. Resume-from-checkpoint logic
        # ==========================================
        last_checkpoint = None
        if os.path.isdir(output_dir):
            last_checkpoint = get_last_checkpoint(output_dir)

        if last_checkpoint is not None:
            print(f"Detected an existing checkpoint ({last_checkpoint}). Resuming training...")
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            print("No checkpoint detected. Starting a fresh training run...")
            train_result = trainer.train(resume_from_checkpoint=False)

        print("Training completed. Preparing final artifacts...")
        # trainer.save_model(str(best_model_dir))
        # tokenizer.save_pretrained(str(best_model_dir))

        # 9. Final evaluation and logging
        print("Running final evaluation...")
        predictions = trainer.predict(tokenized_val)

        save_evaluation_artifacts(
            trainer=trainer,
            train_result=train_result,
            predictions=predictions,
            config=config,
            run_id=run_id,
            results_run_dir=results_run_dir,
            project_root=project_root
        )

        print(f"Experiment {run_id} finished successfully.")

        print("Cleaning up intermediate checkpoint files...")
        try:
            for item in os.listdir(output_dir):
                if item.startswith("checkpoint-"):
                    ckpt_path = os.path.join(output_dir, item)
                    shutil.rmtree(ckpt_path)
            print("Intermediate checkpoint cleanup completed.")
        except Exception as e:
            print(f"Checkpoint cleanup encountered a minor issue: {e}")

    finally:
        # ==========================================
        # 10. Force memory and GPU cache cleanup
        # This block runs whether training finishes normally or exits with an error (such as OOM)
        # ==========================================
        print("Cleaning Python memory and GPU cache...")
        if "model" in locals():
            del model
        if "trainer" in locals():
            del trainer
        if "predictions" in locals():
            del predictions
        if "tokenized_train" in locals():
            del tokenized_train
        if "tokenized_val" in locals():
            del tokenized_val

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("Cleanup finished.")


# Uncomment the block below if you want to test this script directly
# if __name__ == "__main__":
#     # Example project root:
#     run_single_experiment(project_root_path="D:/CSE4601_Text Mining/Yelp_Project")
