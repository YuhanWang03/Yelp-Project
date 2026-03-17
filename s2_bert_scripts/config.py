# Default configuration (used to verify the local pipeline setup)

BASE_CONFIG = {
    "task_type": "5_class",
    "use_regression": False,
    # Recommended options: "roberta-base", "microsoft/deberta-v3-base", "distilbert-base-uncased"
    "model_name": "distilbert-base-uncased",  # Use a smaller model for local testing
    "max_length": 128,                        # Use a shorter sequence length locally to reduce GPU memory usage
    "learning_rate": 2e-5,
    "num_epochs": 1,                          # One epoch is enough for local pipeline testing
    "batch_size": 2,
    "grad_accum_steps": 2,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "linear",
    "fp16": True,                             # Set to False if running on CPU or a GPU without stable half-precision support
    "seed": 42
}
