"""Run the complete purchase-prediction software pipeline.

Every phase keeps its own implementation module. This entry point only calls
those public phase functions in order, writes the requested prediction sample,
and verifies that the expected artifacts exist and are non-empty.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pandas as pd

from src.adaptive import run_phase6
from src.data_prep import run_phase1
from src.eda import run_phase2
from src.evaluate import apply_threshold_correction, run_phase3
from src.explain import run_phase4
from src.insights import run_phase5


PROJECT_ROOT = Path(__file__).resolve().parent


def save_sample_predictions(
    explanation_rows: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save the real high, low, and borderline SHAP examples as predictions."""
    if explanation_rows.empty:
        raise ValueError("Phase 4 returned no explained prediction rows.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve all Phase 4 fields so probabilities, labels, and SHAP drivers
    # remain attached without assuming any source-dataset feature name here.
    explanation_rows.to_csv(path, index=False, float_format="%.8f")
    return path


def validate_generated_artifacts(
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Confirm that all required Phase 1-7 output groups are populated."""
    root = Path(project_root)
    results = root / "results"
    required_files = [
        results / "dataset_summary.txt",
        results / "eda_preprocessing_summary.txt",
        results / "modeling_evaluation_summary.txt",
        results / "shap_explainability_summary.txt",
        results / "business_insights_summary.txt",
        results / "adaptive_experiment_summary.txt",
        results / "tables" / "model_comparison.csv",
        results / "tables" / "model_comparison.txt",
        results / "tables" / "threshold_tuning.csv",
        results / "tables" / "shap_local_examples.csv",
        results / "tables" / "business_insights.csv",
        results / "tables" / "adaptive_before_after.csv",
        results / "predictions" / "sample_predictions_with_explanations.csv",
        root / "models" / "best_model.joblib",
    ]
    missing = [path for path in required_files if not path.is_file()]
    empty = [path for path in required_files if path.is_file() and path.stat().st_size == 0]

    populated_groups = {}
    for name, pattern in {
        "figures": "*.png",
        "tables": "*.csv",
        "predictions": "*.csv",
    }.items():
        files = sorted((results / name).glob(pattern))
        populated_groups[name] = files
        if not files:
            missing.append(results / name / pattern)

    model_files = sorted((root / "models").glob("*.joblib"))
    if not model_files:
        missing.append(root / "models" / "*.joblib")

    if missing or empty:
        messages = [
            *(f"missing: {path}" for path in missing),
            *(f"empty: {path}" for path in empty),
        ]
        raise RuntimeError("Artifact validation failed:\n" + "\n".join(messages))

    return {
        "required_files": required_files,
        "figures": populated_groups["figures"],
        "tables": populated_groups["tables"],
        "predictions": populated_groups["predictions"],
        "models": model_files,
    }


def main() -> None:
    """Execute Phases 1-6 in order and verify the generated project outputs."""
    data_dir = PROJECT_ROOT / "data"
    results_dir = PROJECT_ROOT / "results"
    models_dir = PROJECT_ROOT / "models"

    print("\n[1/6] Data inspection")
    phase1 = run_phase1(data_dir, results_dir / "dataset_summary.txt")
    inspected_shape = phase1["inspection"]["shape"]
    del phase1
    gc.collect()

    print("\n[2/6] Cleaning, preprocessing, and EDA")
    phase2 = run_phase2(data_dir, results_dir / "dataset_summary.txt", results_dir)
    del phase2
    gc.collect()

    print("\n[3/6] Model training, evaluation, and threshold correction")
    phase3 = run_phase3(data_dir, results_dir, models_dir)
    phase3b = apply_threshold_correction(phase3, results_dir, models_dir)
    selected_model = phase3b["best_name"]
    selected_threshold = phase3b["best_threshold"]
    del phase3, phase3b
    gc.collect()

    print("\n[4/6] SHAP explanations")
    phase4 = run_phase4(models_dir / "best_model.joblib", data_dir, results_dir)
    prediction_path = save_sample_predictions(
        phase4["contribution_table"],
        results_dir / "predictions" / "sample_predictions_with_explanations.csv",
    )
    explained_prediction_rows = len(phase4["contribution_table"])
    del phase4
    gc.collect()

    print("\n[5/6] Rule-based business insights")
    phase5 = run_phase5(models_dir / "best_model.joblib", data_dir, results_dir)
    insight_rows = len(phase5["insights"])
    del phase5
    gc.collect()

    print("\n[6/6] Adaptive experiment")
    phase6 = run_phase6(data_dir, results_dir)
    adaptive_mode = phase6["batches"]["mode"]
    retrain_triggered = phase6["triggered"]
    del phase6
    gc.collect()

    artifacts = validate_generated_artifacts(PROJECT_ROOT)
    print("\nFULL PIPELINE COMPLETE")
    print(f"Dataset inspected: {inspected_shape[0]} rows x {inspected_shape[1]} columns")
    print(f"Threshold-aware model: {selected_model}; cutoff={selected_threshold:.6f}")
    print(
        f"Explained sample predictions: {explained_prediction_rows} -> "
        f"{prediction_path.resolve()}"
    )
    print(f"Business-insight rows: {insight_rows}")
    print(f"Adaptive mode: {adaptive_mode}; retraining triggered={retrain_triggered}")
    print(
        "Generated artifacts: "
        f"{len(artifacts['figures'])} figures, {len(artifacts['tables'])} CSV tables, "
        f"{len(artifacts['predictions'])} prediction file(s), "
        f"and {len(artifacts['models'])} model artifact(s)."
    )
    print(f"Results directory: {results_dir.resolve()}")
    print(f"Models directory: {models_dir.resolve()}")


if __name__ == "__main__":
    main()
