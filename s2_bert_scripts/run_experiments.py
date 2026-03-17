import argparse
import traceback
import sys
from config import BASE_CONFIG
from train import run_single_experiment


def main():
    # 1. Define command-line arguments that can be injected externally (for example, from .sh scripts)
    parser = argparse.ArgumentParser(description="Automated Yelp review NLP training script")

    # --- Required argument ---
    parser.add_argument("--project_root", type=str, required=True, help="Path to the project root directory")

    # --- Dynamic hyperparameters (aligned with all keys in config.py) ---
    parser.add_argument("--task_type", type=str, choices=["5_class", "3_class", "binary"])
    parser.add_argument("--use_regression", type=str, choices=["True", "False"])
    parser.add_argument("--model_name", type=str, help="Model name or model path")

    parser.add_argument("--max_length", type=int, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Per-device batch size")
    parser.add_argument("--grad_accum_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--weight_decay", type=float, help="L2 regularization weight")

    parser.add_argument("--warmup_ratio", type=float, help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str, choices=["linear", "cosine", "constant"])

    parser.add_argument("--fp16", type=str, choices=["True", "False"], help="Whether to enable mixed precision")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    # 2. Package incoming command-line arguments into a dictionary to override config.py defaults
    custom_config = {}

    if args.task_type:
        custom_config["task_type"] = args.task_type
    if args.use_regression is not None:
        custom_config["use_regression"] = (args.use_regression == "True")
    if args.model_name:
        custom_config["model_name"] = args.model_name

    if args.max_length is not None:
        custom_config["max_length"] = args.max_length
    if args.learning_rate is not None:
        custom_config["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        custom_config["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        custom_config["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        custom_config["grad_accum_steps"] = args.grad_accum_steps
    if args.weight_decay is not None:
        custom_config["weight_decay"] = args.weight_decay

    if args.warmup_ratio is not None:
        custom_config["warmup_ratio"] = args.warmup_ratio
    if args.lr_scheduler_type:
        custom_config["lr_scheduler_type"] = args.lr_scheduler_type

    if args.fp16 is not None:
        custom_config["fp16"] = (args.fp16 == "True")
    if args.seed is not None:
        custom_config["seed"] = args.seed

    print(f"Received external config overrides: {custom_config}")

    # 3. Launch a single experiment
    try:
        run_single_experiment(custom_config=custom_config, project_root_path=args.project_root)
        print("Experiment finished successfully.")
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        traceback.print_exc()
        # Return a non-zero exit code to the shell script without breaking the full outer loop
        sys.exit(1)


if __name__ == "__main__":
    main()
