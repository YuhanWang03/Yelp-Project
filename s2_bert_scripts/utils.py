import os
import json
import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)


def setup_experiment_paths(config, project_root):
    lvl1_task = config["task_type"]
    lvl2_mode = "Regression" if config["use_regression"] else "Classification"
    lvl3_model = config["model_name"].split("/")[-1]

    model_base_dir = project_root / "s2_bert_models" / lvl1_task / lvl2_mode / lvl3_model
    model_base_dir.mkdir(parents=True, exist_ok=True)

    existing_runs = [d.name for d in model_base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not existing_runs:
        run_id = "run_001"
    else:
        max_id = max([int(name.split("_")[1]) for name in existing_runs])
        run_id = f"run_{max_id + 1:03d}"

    print(f"Current experiment: {lvl1_task} | {lvl2_mode} | {lvl3_model} | {run_id}")

    model_run_dir = model_base_dir / run_id
    output_dir = model_run_dir / "checkpoints"
    best_model_dir = model_run_dir / "best_model"
    results_run_dir = project_root / "s2_bert_results" / lvl1_task / lvl2_mode / lvl3_model / run_id
    logging_dir = results_run_dir / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    with open(results_run_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    return model_run_dir, output_dir, best_model_dir, logging_dir, results_run_dir, run_id


def get_compute_metrics_fn(config):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if config["use_regression"]:
            preds = np.squeeze(logits)
            mse = mean_squared_error(labels, preds)
            mae = mean_absolute_error(labels, preds)

            # During training, also report rounded classification-style metrics for easier monitoring
            if config["task_type"] == "binary":
                rounded_preds = (preds >= 0.5).astype(int)
            elif config["task_type"] == "3_class":
                rounded_preds = np.clip(np.round(preds), 0, 2)
            else:
                rounded_preds = np.clip(np.round(preds), 1, 5)

            acc = accuracy_score(labels, rounded_preds)
            macro_f1 = f1_score(labels, rounded_preds, average="macro")
            return {"mse": mse, "mae": mae, "pseudo_accuracy": acc, "pseudo_macro_f1": macro_f1}
        else:
            preds = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, preds)
            macro_f1 = f1_score(labels, preds, average="macro")
            return {"accuracy": acc, "macro_f1": macro_f1}

    return compute_metrics


def save_evaluation_artifacts(trainer, train_result, predictions, config, run_id, results_run_dir, project_root):
    logits = predictions.predictions
    true_labels = predictions.label_ids

    # =========================================
    # Critical safety check: detect NaN outputs caused by unstable training
    # =========================================
    if np.isnan(logits).any():
        print(f"Warning: experiment {run_id} produced NaN outputs.")
        print("This run will be logged as failed (NaN), and figure generation will be skipped.")

        log_row = {
            "Run_ID": run_id,
            "Task_Type": config["task_type"],
            "Mode": "Regression" if config["use_regression"] else "Classification",
            "Model": config["model_name"].split("/")[-1],
            "Epochs": config["num_epochs"],
            "Batch_Size": config["batch_size"],
            "LR": config["learning_rate"],
            "Accuracy": "NaN",
            "Macro_F1": "NaN",
            "AUC": "NaN",
            "Off_By_1_Acc": "NaN",
            "Adj_Error_Ratio": "NaN",
            "Mid_Confusion_Ratio": "NaN",
            "MSE": "NaN",
            "MAE": "NaN",
            "Train_Time(s)": "NaN",
            "Train_Speed(s/s)": "NaN",
            "Eval_Time(s)": "NaN",
            "Eval_Speed(s/s)": "NaN"
        }

        results_log_path = project_root / "s2_bert_results" / "experiments_log.csv"
        models_log_path = project_root / "s2_bert_models" / "models_registry.csv"

        for filepath in [results_log_path, models_log_path]:
            file_exists = filepath.exists()
            with open(filepath, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=log_row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(log_row)
        return

    # =========================================
    # Continue with the normal evaluation flow if no NaNs are detected
    # =========================================
    preds = None
    pred_labels = None

    lvl1_task = config["task_type"]
    lvl2_mode = "Regression" if config["use_regression"] else "Classification"
    lvl3_model = config["model_name"].split("/")[-1]

    # 1. Extract efficiency metrics
    train_metrics = train_result.metrics
    eval_metrics = predictions.metrics

    train_time = round(train_metrics.get("train_runtime", 0), 2)
    train_throughput = round(train_metrics.get("train_samples_per_second", 0), 2)
    eval_time = round(eval_metrics.get("test_runtime", 0), 2)
    eval_throughput = round(eval_metrics.get("test_samples_per_second", 0), 2)

    print("\n--- Model Efficiency ---")
    print(f"Train Time: {train_time} sec | Train Throughput: {train_throughput} samples/sec")

    # 2. Plot learning curves
    log_history = trainer.state.log_history
    train_epochs = [x["epoch"] for x in log_history if "loss" in x]
    train_losses = [x["loss"] for x in log_history if "loss" in x]
    eval_epochs = [x["epoch"] for x in log_history if "eval_loss" in x]
    eval_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]

    if train_losses and eval_losses:
        plt.figure(figsize=(8, 5))
        plt.plot(train_epochs, train_losses, label="Training Loss", marker="o")
        plt.plot(eval_epochs, eval_losses, label="Validation Loss", marker="s")
        plt.title(f"Learning Curves\n{lvl3_model} | {run_id}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(results_run_dir / "learning_curves.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 3. Parse predictions and print a summary report
    if config["use_regression"]:
        preds = np.squeeze(logits)
        if config["task_type"] == "binary":
            pred_labels = (preds >= 0.5).astype(int)
            target_names = ["Negative (0-2)", "Positive (4-5)"]
        elif config["task_type"] == "3_class":
            pred_labels = np.clip(np.round(preds), 0, 2)
            target_names = ["Negative (0)", "Neutral (1)", "Positive (2)"]
        else:
            pred_labels = np.clip(np.round(preds), 1, 5)
            target_names = ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
    else:
        pred_labels = np.argmax(logits, axis=-1)
        if config["task_type"] == "binary":
            target_names = ["Negative", "Positive"]
        elif config["task_type"] == "3_class":
            target_names = ["Negative (1-2)", "Neutral (3)", "Positive (4-5)"]
        else:
            target_names = ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]

    print("\n--- Classification Report ---")
    print(classification_report(true_labels, pred_labels, target_names=target_names))

    # 4. Plot the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix\n{lvl3_model} | {lvl2_mode} | {run_id}", fontsize=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(results_run_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. For binary classification only: plot the ROC curve and compute AUC
    log_auc = "N/A"
    if config["task_type"] == "binary" and not config["use_regression"]:
        probs = softmax(logits, axis=1)[:, 1]
        fpr, tpr, thresholds = roc_curve(true_labels, probs)
        roc_auc = auc(fpr, tpr)
        log_auc = round(roc_auc, 4)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver Operating Characteristic\n{lvl3_model} | {run_id}")
        plt.legend(loc="lower right")
        plt.savefig(results_run_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 6. Additional error analysis for 5-class tasks (works for both classification and regression)
    off_by_1_acc, adj_error_ratio, mid_confusion_ratio = "N/A", "N/A", "N/A"

    if config["task_type"] == "5_class":
        if config["use_regression"]:
            y_true_stars = np.round(true_labels)
            y_pred_stars = pred_labels
        else:
            y_true_stars = true_labels + 1
            y_pred_stars = pred_labels + 1

        total_samples = len(y_true_stars)
        errors_mask = y_true_stars != y_pred_stars
        total_errors = np.sum(errors_mask)

        if total_errors > 0:
            off_by_1_mask = np.abs(y_true_stars - y_pred_stars) <= 1
            off_by_1_acc = round(np.sum(off_by_1_mask) / total_samples, 4)

            adjacent_errors_mask = np.abs(y_true_stars - y_pred_stars) == 1
            adj_error_ratio = round(np.sum(adjacent_errors_mask) / total_errors, 4)

            middle_adj_mask = (
                ((y_true_stars == 2) & (y_pred_stars == 3)) |
                ((y_true_stars == 3) & (y_pred_stars == 2)) |
                ((y_true_stars == 3) & (y_pred_stars == 4)) |
                ((y_true_stars == 4) & (y_pred_stars == 3))
            )
            mid_confusion_ratio = round(np.sum(middle_adj_mask) / total_errors, 4)

    # =========================================
    # 7. Append results to the global experiment log and model registry
    # =========================================
    if config["use_regression"]:
        # Regression reports MSE and MAE, and also includes rounded Accuracy/F1 for easier comparison
        log_mse = round(mean_squared_error(true_labels, preds), 4)
        log_mae = round(mean_absolute_error(true_labels, preds), 4)
        log_acc = round(accuracy_score(true_labels, pred_labels), 4)
        log_f1 = round(f1_score(true_labels, pred_labels, average="macro"), 4)
    else:
        # Classification reports Accuracy/F1 directly; MSE and MAE are not applicable
        log_mse, log_mae = "N/A", "N/A"
        log_acc = round(accuracy_score(true_labels, pred_labels), 4)
        log_f1 = round(f1_score(true_labels, pred_labels, average="macro"), 4)

    log_row = {
        "Run_ID": run_id,
        "Task_Type": config["task_type"],
        "Mode": lvl2_mode,
        "Model": lvl3_model,
        "Epochs": config["num_epochs"],
        "Batch_Size": config["batch_size"],
        "LR": config["learning_rate"],
        "Accuracy": log_acc,
        "Macro_F1": log_f1,
        "AUC": log_auc,
        "Off_By_1_Acc": off_by_1_acc,
        "Adj_Error_Ratio": adj_error_ratio,
        "Mid_Confusion_Ratio": mid_confusion_ratio,
        "MSE": log_mse,
        "MAE": log_mae,
        "Train_Time(s)": train_time,
        "Train_Speed(s/s)": train_throughput,
        "Eval_Time(s)": eval_time,
        "Eval_Speed(s/s)": eval_throughput
    }

    (project_root / "s2_bert_results").mkdir(parents=True, exist_ok=True)
    (project_root / "s2_bert_models").mkdir(parents=True, exist_ok=True)

    results_log_path = project_root / "s2_bert_results" / "experiments_log.csv"
    models_log_path = project_root / "s2_bert_models" / "models_registry.csv"

    def write_to_csv(filepath, data_row):
        file_exists = filepath.exists()
        with open(filepath, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_row)

    write_to_csv(results_log_path, log_row)
    write_to_csv(models_log_path, log_row)

    print("-" * 50)
    print(f"Experiment results appended to: {results_log_path}")
