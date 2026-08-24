"""Phase 3 evaluation, figure generation, and best-model persistence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)

from src.models import (
    ThresholdedClassifier,
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


def tune_model_thresholds(
    trained_models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    default_comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Find each held-out probability threshold that maximizes positive-class F1."""
    default_by_model = default_comparison.set_index("model")
    rows = []
    details = {}
    for name, model in trained_models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        default_predictions = model.predict(X_test)
        precision_values, recall_values, thresholds = precision_recall_curve(
            y_test, probabilities
        )
        # precision/recall contain one final point with no corresponding threshold.
        denominator = precision_values[:-1] + recall_values[:-1]
        f1_values = np.divide(
            2.0 * precision_values[:-1] * recall_values[:-1],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        best_index = int(np.argmax(f1_values))
        tuned_threshold = float(thresholds[best_index])
        tuned_predictions = (probabilities >= tuned_threshold).astype(int)
        default_matrix = confusion_matrix(y_test, default_predictions)
        tuned_matrix = confusion_matrix(y_test, tuned_predictions)
        row = {
            "model": name,
            "roc_auc": float(default_by_model.loc[name, "roc_auc"]),
            "default_threshold": 0.5,
            "default_accuracy": accuracy_score(y_test, default_predictions),
            "default_precision": precision_score(y_test, default_predictions, zero_division=0),
            "default_recall": recall_score(y_test, default_predictions, zero_division=0),
            "default_f1": f1_score(y_test, default_predictions, zero_division=0),
            "tuned_threshold": tuned_threshold,
            "tuned_accuracy": accuracy_score(y_test, tuned_predictions),
            "tuned_precision": precision_score(y_test, tuned_predictions, zero_division=0),
            "tuned_recall": recall_score(y_test, tuned_predictions, zero_division=0),
            "tuned_f1": f1_score(y_test, tuned_predictions, zero_division=0),
        }
        rows.append(row)
        details[name] = {
            "probabilities": probabilities,
            "default_predictions": default_predictions,
            "tuned_predictions": tuned_predictions,
            "default_confusion_matrix": default_matrix,
            "tuned_confusion_matrix": tuned_matrix,
            "threshold": tuned_threshold,
        }
    table = (
        pd.DataFrame(rows)
        .sort_values(["tuned_f1", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    return table, details


def save_tuned_confusion_matrices(
    trained_models: dict[str, Any],
    tuning_details: dict[str, dict[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """Save held-out confusion matrices using each F1-maximizing threshold."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, model in trained_models.items():
        detail = tuning_details[name]
        display = ConfusionMatrixDisplay(
            confusion_matrix=detail["tuned_confusion_matrix"],
            display_labels=model.named_steps["classifier"].classes_,
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        display.plot(ax=ax, cmap="Greens", colorbar=False, values_format=",d")
        ax.set_title(
            f"Tuned Confusion Matrix - {name}\n"
            f"threshold={detail['threshold']:.4f}, held-out test set"
        )
        fig.tight_layout()
        safe_name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        path = destination / f"confusion_matrix_{safe_name}_tuned.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def apply_threshold_correction(
    phase3_results: dict[str, Any],
    results_dir: str | Path = "results",
    models_dir: str | Path = "models",
) -> dict[str, Any]:
    """Tune thresholds, reselect the best model, and update Phase 3 artifacts."""
    result_path = Path(results_dir)
    trained = phase3_results["trained_models"]
    data = phase3_results["data"]
    comparison = phase3_results["comparison"]
    tuning, tuning_details = tune_model_thresholds(
        trained, data["X_test"], data["y_test"], comparison
    )
    threshold_path = result_path / "tables" / "threshold_tuning.csv"
    tuning.to_csv(threshold_path, index=False, float_format="%.6f")
    tuned_confusion_paths = save_tuned_confusion_matrices(
        trained, tuning_details, result_path / "figures"
    )

    # Highest tuned F1 wins; held-out ROC-AUC is the deterministic tiebreaker.
    best_name = str(tuning.iloc[0]["model"])
    best_threshold = float(tuning.iloc[0]["tuned_threshold"])
    thresholded_model = ThresholdedClassifier(trained[best_name], best_threshold)
    artifact_path = Path(models_dir) / "best_model.joblib"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(thresholded_model, artifact_path, compress=3)

    readable_path = result_path / "tables" / "model_comparison.txt"
    default_text = comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}")
    tuned_text = tuning.to_string(index=False, float_format=lambda value: f"{value:.4f}")
    readable_path.write_text(
        "PHASE 3 DEFAULT-THRESHOLD COMPARISON\n"
        "F1 and ROC-AUC are primary; accuracy is secondary because the target is imbalanced.\n\n"
        + default_text
        + "\n\nPHASE 3B F1-MAXIMIZING THRESHOLD CORRECTION\n"
        "Thresholds were selected from each held-out precision-recall curve.\n"
        "Models are ranked by tuned F1, with ROC-AUC as the tiebreaker.\n\n"
        "The same held-out set selects and reports the cutoff as requested; a production workflow should use a separate validation fold.\n\n"
        + tuned_text
        + f"\n\nSelected model: {best_name}\nSelected threshold: {best_threshold:.6f}\n",
        encoding="utf-8",
    )

    summary_path = result_path / "modeling_evaluation_summary.txt"
    existing = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    base = existing.split("\nPHASE 3B - THRESHOLD CORRECTION", maxsplit=1)[0].rstrip()
    correction = (
        "\n\nPHASE 3B - THRESHOLD CORRECTION\n"
        "================================\n"
        "The default 0.5 cutoff produced misleadingly different class trade-offs despite similar ROC-AUC.\n"
        "Each threshold now maximizes positive-class F1 on the requested held-out precision-recall curve.\n\n"
        "Caveat: a production workflow should tune on validation data and keep the test set for one final evaluation.\n\n"
        + tuned_text
        + f"\n\nBest model by tuned F1 (ROC-AUC tiebreak): {best_name}\n"
        + f"Operational threshold stored in artifact: {best_threshold:.6f}\n"
    )
    summary_path.write_text(base + correction, encoding="utf-8")

    print("\nPhase 3b threshold tuning (F1-maximizing cutoffs):")
    print(tuned_text)
    print(f"\nTuned-threshold best model: {best_name} at {best_threshold:.6f}")
    print(f"Threshold-aware artifact saved to: {artifact_path.resolve()}")
    return {
        "tuning": tuning,
        "tuning_details": tuning_details,
        "threshold_path": threshold_path,
        "tuned_confusion_paths": tuned_confusion_paths,
        "best_name": best_name,
        "best_threshold": best_threshold,
        "artifact_path": artifact_path,
        "summary_path": summary_path,
    }


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
    phase3 = run_phase3()
    apply_threshold_correction(phase3)
