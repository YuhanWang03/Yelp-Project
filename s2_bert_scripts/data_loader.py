import pandas as pd
from datasets import Dataset


def load_and_preprocess_data(train_path, val_path, task_type, use_regression):
    """
    Load CSV data, then filter and map labels based on task_type and use_regression.
    Returns processed Hugging Face Dataset objects and the required num_labels value.
    """
    print(f"Loading data from: {train_path.parent} ...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Fill missing values and make sure the text column is stored as strings
    train_df["text"] = train_df["text"].fillna("").astype(str)
    val_df["text"] = val_df["text"].fillna("").astype(str)

    # ==========================================
    # Step A: Filter data and apply the base label mapping for each task type
    # ==========================================
    if task_type == "binary":
        # Remove 3-star reviews; map 1-2 stars to negative (0) and 4-5 stars to positive (1)
        train_df = train_df[train_df["stars"] != 3].copy()
        val_df = val_df[val_df["stars"] != 3].copy()
        train_df["label"] = train_df["stars"].apply(lambda x: 0 if x < 3 else 1)
        val_df["label"] = val_df["stars"].apply(lambda x: 0 if x < 3 else 1)
        num_classes = 2

    elif task_type == "3_class":
        # Map 1-2 stars to negative (0), 3 stars to neutral (1), and 4-5 stars to positive (2)
        def map_3_class(stars):
            return 0 if stars < 3 else (1 if stars == 3 else 2)

        train_df["label"] = train_df["stars"].apply(map_3_class)
        val_df["label"] = val_df["stars"].apply(map_3_class)
        num_classes = 3

    elif task_type == "5_class":
        if use_regression:
            # For regression, use the original star rating directly (1-5)
            train_df["label"] = train_df["stars"]
            val_df["label"] = val_df["stars"]
        else:
            # For classification, shift labels to start from 0 (0-4)
            train_df["label"] = train_df["stars"] - 1
            val_df["label"] = val_df["stars"] - 1
        num_classes = 5

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    # ==========================================
    # Step B: Set the label dtype based on use_regression
    # ==========================================
    if use_regression:
        train_df["label"] = train_df["label"].astype(float)
        val_df["label"] = val_df["label"].astype(float)
        num_labels = 1  # Regression outputs a single continuous value
    else:
        train_df["label"] = train_df["label"].astype(int)
        val_df["label"] = val_df["label"].astype(int)
        num_labels = num_classes  # Classification outputs one node per class

    # Convert to Hugging Face Dataset format (preserve_index=False avoids carrying over a useless index)
    train_dataset = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)

    print(f"Data ready: {task_type} | Regression: {use_regression} | Model head size: {num_labels}")
    return train_dataset, val_dataset, num_labels


def tokenize_data(train_dataset, val_dataset, tokenizer, max_length):
    """
    Batch-tokenize the datasets with the provided tokenizer and convert them to PyTorch tensor format.
    """
    print("Tokenizing datasets...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    # batched=True enables faster tokenization
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    # Remove the raw text column and keep only model inputs plus labels
    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_val = tokenized_val.remove_columns(["text"])

    # Set the format to PyTorch tensors
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")

    print("Tokenization completed.")
    return tokenized_train, tokenized_val
