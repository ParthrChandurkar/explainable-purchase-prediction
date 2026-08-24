"""Phase 3 evaluation, figure generation, and best-model persistence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models import (
    audit_existing_encoding_decisions,
    build_model_pipelines,
    prepare_training_data,
    train_models,
    verify_corrected_preprocessing,
)


def evaluate_trained_models(
    trained_models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Compute every requested metric from real held-out predictions."""
    rows = []
    prediction_details = {}
    for name, model in trained_models.items():
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }
        # F1 and ROC-AUC are equally weighted because class imbalance makes
        # accuracy a secondary metric for model selection.
        metrics["primary_score"] = (metrics["f1"] + metrics["roc_auc"]) / 2.0
        rows.append(metrics)
        prediction_details[name] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "confusion_matrix": confusion_matrix(y_test, predictions),
        }
    comparison = (
        pd.DataFrame(rows)
        .sort_values(["primary_score", "f1", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    return comparison, prediction_details


def save_confusion_matrices(
    trained_models: dict[str, Any],
    prediction_details: dict[str, dict[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """Save one held-out confusion matrix plot per trained model."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, model in trained_models.items():
        matrix = prediction_details[name]["confusion_matrix"]
        labels = model.named_steps["classifier"].classes_
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        display.plot(ax=ax, cmap="Blues", colorbar=False, values_format=",d")
        ax.set_title(f"Confusion Matrix - {name}\nHeld-out stratified test set")
        fig.tight_layout()
        safe_name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        path = destination / f"confusion_matrix_{safe_name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def save_comparison(
    comparison: pd.DataFrame,
    tables_dir: str | Path,
) -> tuple[Path, Path]:
    """Save machine-readable and aligned text versions of the model table."""
    destination = Path(tables_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "model_comparison.csv"
    text_path = destination / "model_comparison.txt"
    comparison.to_csv(csv_path, index=False, float_format="%.6f")
    readable = comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}")
    text_path.write_text(
        "Primary comparison metrics: F1 and ROC-AUC. Accuracy is secondary because the target is imbalanced.\n\n"
        + readable
        + "\n",
        encoding="utf-8",
    )
    return csv_path, text_path


def save_best_model(
    comparison: pd.DataFrame,
    trained_models: dict[str, Any],
    output_dir: str | Path,
) -> tuple[str, Path]:
    """Persist the pipeline with the highest F1/ROC-AUC composite score."""
    best_name = str(comparison.iloc[0]["model"])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "best_model.joblib"
    joblib.dump(trained_models[best_name], path, compress=3)
    return best_name, path


def render_modeling_summary(
    audit: pd.DataFrame,
    feature_counts: dict[str, int],
    data: dict[str, Any],
    comparison: pd.DataFrame,
    best_name: str,
    artifact_path: Path,
    confusion_paths: list[Path],
) -> str:
    """Create a concise readable record of actual Phase 3 decisions/results."""
    lines = [
        "PHASE 3 - MODELING AND EVALUATION SUMMARY",
        "=" * 41,
        f"Fixed target: {data['target']}",
        f"Training rows: {len(data['X_train'])}",
        f"Test rows: {len(data['X_test'])}",
        "Split: stratified, test_size=0.20, random_state=42",
        "",
        "HIGH-CARDINALITY ENCODING AUDIT (COMPLETED BEFORE TRAINING)",
        "-----------------------------------------------------------",
        audit.to_string(index=False),
        "",
        f"Corrected tree feature count: {feature_counts['tree']}",
        f"Corrected logistic feature count: {feature_counts['linear']}",
        "XGBoost skipped: it is outside the project-wide allowed dependency list; no third model was forced.",
        "",
        "MODEL COMPARISON",
        "----------------",
        "F1 and ROC-AUC are primary; accuracy is secondary because the target is imbalanced.",
        comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        f"Best model by mean(F1, ROC-AUC): {best_name}",
        f"Saved pipeline: {artifact_path.resolve()}",
        "Confusion matrices:",
        *[f"- {path.resolve()}" for path in confusion_paths],
    ]
    return "\n".join(lines) + "\n"


def run_phase3(
    data_dir: str | Path = "data",
    results_dir: str | Path = "results",
    models_dir: str | Path = "models",
) -> dict[str, Any]:
    """Audit encoding, train both models, evaluate, save, and print results."""
    result_path = Path(results_dir)
    data = prepare_training_data(data_dir, result_path / "dataset_summary.txt")

    audit = audit_existing_encoding_decisions(
        data["cleaned"],
        result_path / "tables" / "encoding_decisions.csv",
        result_path / "tables" / "encoding_correction_audit.csv",
    )
    print("High-cardinality one-hot audit completed before training:")
    print(audit.to_string(index=False))

    # Persist the corrected Phase 2 roles so future phases reuse the fix.
    pd.DataFrame(data["roles"]["decisions"]).to_csv(
        result_path / "tables" / "encoding_decisions.csv", index=False
    )
    feature_counts = verify_corrected_preprocessing(data["X_train"], data["roles"])
    print(f"Corrected feature counts: {feature_counts} (must remain below 300)")

    pipelines = build_model_pipelines(data["roles"])
    trained = train_models(pipelines, data["X_train"], data["y_train"])
    comparison, prediction_details = evaluate_trained_models(
        trained, data["X_test"], data["y_test"]
    )
    comparison_paths = save_comparison(comparison, result_path / "tables")
    confusion_paths = save_confusion_matrices(
        trained, prediction_details, result_path / "figures"
    )
    best_name, artifact_path = save_best_model(comparison, trained, models_dir)

    summary = render_modeling_summary(
        audit,
        feature_counts,
        data,
        comparison,
        best_name,
        artifact_path,
        confusion_paths,
    )
    summary_path = result_path / "modeling_evaluation_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print("\nFinal model comparison (F1 and ROC-AUC are primary; accuracy is secondary):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nBest model saved: {best_name} -> {artifact_path.resolve()}")
    return {
        "data": data,
        "audit": audit,
        "feature_counts": feature_counts,
        "trained_models": trained,
        "comparison": comparison,
        "prediction_details": prediction_details,
        "comparison_paths": comparison_paths,
        "confusion_paths": confusion_paths,
        "best_name": best_name,
        "artifact_path": artifact_path,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    run_phase3()
