import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

# ─────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────
EXPERIMENT_NAME = "hand-gesture-recognition"
REGISTRY_MODEL_NAME = "hand-gesture-svm-rbf"


# ─────────────────────────────────────────────
# 1. Experiment setup
# ─────────────────────────────────────────────
def setup_experiment(experiment_name: str = EXPERIMENT_NAME) -> str:
    """
    Create (or retrieve) an MLflow experiment.
    Returns the experiment_id.
    """
    # mlflow.set_tracking_uri("mlruns")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"[MLflow] Created experiment '{experiment_name}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"[MLflow] Using existing experiment '{experiment_name}' (id={experiment_id})")
    mlflow.set_experiment(experiment_name)
    return experiment_id


# ─────────────────────────────────────────────
# 2. Dataset logging
# ─────────────────────────────────────────────
def log_dataset(df: pd.DataFrame, dataset_path: str = "hand_landmarks_data.csv") -> None:
    """
    Log dataset statistics as params and a summary .txt artifact.
    Must be called inside an active mlflow.start_run() context.
    """
    mlflow.log_param("dataset_path", dataset_path)
    mlflow.log_param("dataset_n_samples", len(df))
    mlflow.log_param("dataset_n_features", df.shape[1] - 1)
    mlflow.log_param("dataset_n_classes", df["label"].nunique())

    for cls, cnt in df["label"].value_counts().to_dict().items():
        mlflow.log_param(f"class_count_{cls}", cnt)

    with tempfile.TemporaryDirectory() as tmp:
        summary_path = os.path.join(tmp, "dataset_summary.txt")
        with open(summary_path, "w") as fh:
            fh.write(f"Dataset : {dataset_path}\n")
            fh.write(f"Samples : {len(df)}\n")
            fh.write(f"Features: {df.shape[1] - 1}\n")
            fh.write(f"Classes : {df['label'].nunique()}\n\n")
            fh.write("Class distribution:\n")
            fh.write(df["label"].value_counts().to_string())
            fh.write("\n\nDescriptive statistics:\n")
            fh.write(df.describe().to_string())
        mlflow.log_artifact(summary_path, artifact_path="dataset")


# ─────────────────────────────────────────────
# 3. Parameters logging
# ─────────────────────────────────────────────
def log_parameters(params: dict) -> None:
    """Log a flat dictionary of hyper-parameters."""
    mlflow.log_params(params)


# ─────────────────────────────────────────────
# 4. Metrics logging
# ─────────────────────────────────────────────
def compute_and_log_metrics(y_true, y_pred, split: str = "test") -> dict:
    """
    Compute accuracy / precision / recall / F1 (macro) from already-produced
    predictions and log them to the active run.
    Returns the metrics dict.
    """
    metrics = {
        f"{split}_accuracy" : accuracy_score(y_true, y_pred),
        f"{split}_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        f"{split}_recall"   : recall_score(y_true, y_pred, average="macro", zero_division=0),
        f"{split}_f1"       : f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    mlflow.log_metrics(metrics)
    return metrics


# ─────────────────────────────────────────────
# 5. Artifact helpers
# ─────────────────────────────────────────────
def log_confusion_matrix(y_true, y_pred, class_names=None, split: str = "test") -> None:
    """Save and log a confusion matrix heatmap PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix — {split} set")
    plt.tight_layout()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"confusion_matrix_{split}.png")
        fig.savefig(path, dpi=150)
        mlflow.log_artifact(path, artifact_path="plots")
    plt.close(fig)


def log_classification_report(y_true, y_pred, class_names=None, split: str = "test") -> None:
    """Save and log a per-class text classification report."""
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"classification_report_{split}.txt")
        with open(path, "w") as fh:
            fh.write(report)
        mlflow.log_artifact(path, artifact_path="reports")


def log_model_comparison_chart(comparison_df: pd.DataFrame) -> None:
    """
    Log a grouped bar chart comparing models across Accuracy / Precision / Recall / F1.
    `comparison_df` must have columns: Model, Accuracy, Precision, Recall, F1
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, row in comparison_df.reset_index(drop=True).iterrows():
        offset = (idx - len(comparison_df) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, row[metrics].values, width,
            label=row["Model"], color=colors[idx % len(colors)],
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.95, 1.005)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Hand Gesture Recognition")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model_comparison.png")
        fig.savefig(path, dpi=150)
        mlflow.log_artifact(path, artifact_path="plots")
    plt.close(fig)
    print("[MLflow] Model comparison chart logged.")


# ─────────────────────────────────────────────
# 6. Model artifact logging
# ─────────────────────────────────────────────
def log_sklearn_model(model, run_name: str) -> mlflow.models.model.ModelInfo:
    """Log an already-fitted sklearn-compatible model as an MLflow artifact."""
    return mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=f"model_{run_name}",
        registered_model_name=None,   # register separately via register_best_model
    )


# ─────────────────────────────────────────────
# 7. Full single-model logging run
# ─────────────────────────────────────────────
def log_run(
    fitted_model,           # already trained model object from the notebook
    run_name: str,
    params: dict,           # hyper-params to record (extracted from model.get_params())
    val_pred,   y_val,      # predictions & ground-truth for the validation split
    test_pred,  y_test,     # predictions & ground-truth for the test split
    df_raw: pd.DataFrame,   # original (pre-processing) dataframe for dataset stats
    class_names=None,
) -> dict:
    """
    Open an MLflow run and log everything for one already-fitted model:
      1. Dataset statistics
      2. Hyper-parameters
      3. Val + test metrics  (computed from the predictions you pass in)
      4. Confusion matrices & classification reports
      5. The fitted model artifact

    No training happens here — pass the model after you've called .fit() in the notebook.
    Returns a dict with test metrics + run_id for downstream use.
    """
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n[MLflow] ▶  Run: '{run_name}'  (run_id={run.info.run_id})")

        log_dataset(df_raw)
        log_parameters(params)

        val_metrics  = compute_and_log_metrics(y_val,  val_pred,  split="val")
        test_metrics = compute_and_log_metrics(y_test, test_pred, split="test")

        log_confusion_matrix(y_val,  val_pred,  class_names, split="val")
        log_confusion_matrix(y_test, test_pred, class_names, split="test")
        log_classification_report(y_val,  val_pred,  class_names, split="val")
        log_classification_report(y_test, test_pred, class_names, split="test")

        log_sklearn_model(fitted_model, run_name)

        print(f"[MLflow]    test | acc={test_metrics['test_accuracy']:.4f}"
              f"  prec={test_metrics['test_precision']:.4f}"
              f"  rec={test_metrics['test_recall']:.4f}"
              f"  f1={test_metrics['test_f1']:.4f}")

    return {**test_metrics, "run_id": run.info.run_id, "model_name": run_name}


# ─────────────────────────────────────────────
# 8. Comparison run (call after all model runs)
# ─────────────────────────────────────────────
def log_comparison_run(results: list) -> pd.DataFrame:
    """
    Open a dedicated 'model-comparison' run, log the grouped bar chart and a CSV table.
    `results` — list of dicts returned by log_run().
    """
    comparison_df = pd.DataFrame([
        {
            "Model"    : r["model_name"],
            "Accuracy" : r["test_accuracy"],
            "Precision": r["test_precision"],
            "Recall"   : r["test_recall"],
            "F1"       : r["test_f1"],
        }
        for r in results
    ])
    print("\nModel Comparison Table:")
    print(comparison_df.to_string(index=False))

    with mlflow.start_run(run_name="model-comparison"):
        log_model_comparison_chart(comparison_df)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "model_comparison.csv")
            comparison_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_path="reports")

    print("[MLflow] Comparison run complete.")
    return comparison_df


# ─────────────────────────────────────────────
# 9. Model Registry
# ─────────────────────────────────────────────
def register_best_model(
    run_id: str,
    artifact_path: str,
    registry_name: str = REGISTRY_MODEL_NAME,
    description: str = "Best hand-gesture classifier — SVM RBF kernel",
) -> None:
    """
    Register a logged model artifact in the MLflow Model Registry
    and transition it to the 'Staging' stage.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri=model_uri, name=registry_name)
    print(f"[MLflow] Registered '{registry_name}' v{mv.version} from run {run_id}")

    client = MlflowClient()
    client.update_registered_model(name=registry_name, description=description)
    client.update_model_version(
        name=registry_name,
        version=mv.version,
        description=(
            "SVM with RBF kernel (C=375) trained on normalized hand landmark features. "
            "Achieves the highest test accuracy and F1 among SVM, RandomForest, and XGBoost."
        ),
    )
    client.transition_model_version_stage(
        name=registry_name,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    print(f"[MLflow] Model '{registry_name}' v{mv.version} → Staging")


# ─────────────────────────────────────────────
# 10. Load registered model
# ─────────────────────────────────────────────
def load_registered_model(registry_name: str = REGISTRY_MODEL_NAME, stage: str = "Staging"):
    """Load the registered model from the MLflow registry."""
    model_uri = f"models:/{registry_name}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"[MLflow] Loaded '{registry_name}' from stage '{stage}'")
    return model
